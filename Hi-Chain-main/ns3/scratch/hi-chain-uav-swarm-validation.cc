/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Hi-Chain UAV swarm validation + cache replacement (NS-3 v3.25)
 *
 * Paper-aligned defaults: hi-chain-paper-params.h
 * Cache policies: none | lru | freq | rl (RL actions from Python export)
 *
 * RL workflow (BDQN is PyTorch — not embedded in NS-3):
 *   1. python train_model007.py  (save results/train/bdqn_policy.pt)
 *   2. python export_ns3_rl_trace.py --n-nodes <same as nUavs>
 *   3. ./waf --run "scratch/hi-chain-uav-swarm-validation --cachePolicy=rl
 *        --accessTrace=.../ns3_access_trace.csv --rlActions=.../ns3_rl_actions.csv"
 */

#include "hi-chain-paper-params.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/random-variable-stream.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("HiChainUavSwarmValidation");

static const uint16_t g_port = 9900;

enum HitType : uint8_t
{
    HIT_MISS = 0,
    HIT_LOCAL = 1,
    HIT_REMOTE = 2,
    HIT_RECOVERY = 3
};

enum PacketType : uint8_t
{
    PKT_BLOCK_REQUEST = 1,
    PKT_BLOCK_RESPONSE = 2,
    PKT_FRAGMENT_OFFER = 3
};

enum CachePolicy
{
    CACHE_NONE = 0,
    CACHE_LRU = 1,
    CACHE_FREQ = 2,
    CACHE_RL = 3
};

#pragma pack(push, 1)
struct HiChainPacket
{
    uint8_t type;
    uint32_t blockId;
    uint32_t srcNodeId;
    uint32_t reqSeq;
};
#pragma pack(pop)

struct AccessRecord
{
    uint32_t nodeId;
    uint32_t blockId;
    double latencyMs;
    HitType hitType;
};

struct RlActionRow
{
    uint32_t step;
    uint32_t evictSlot;
    uint32_t addBlockId;
};

struct BlockMeta
{
    double frequency;
    uint32_t kRecover;
};

static std::vector<AccessRecord> g_records;
class NodeCacheState;
class HiChainBlockAccessApp;
static std::map<uint32_t, std::shared_ptr<NodeCacheState>> g_nodeState;
static std::vector<Ptr<HiChainBlockAccessApp>> g_accessApps;
static std::vector<uint32_t> g_accessTrace;
static std::vector<RlActionRow> g_rlActions;
static std::map<uint32_t, BlockMeta> g_blockMeta;
static uint32_t g_numBlocks = 0;
static uint32_t g_blockSizeBytes = 0;
static uint32_t g_cacheCapacityBytes = 0;
static double g_cacheHighWatermark = 0.9;
static uint32_t g_kRecoverGlobal = 20;
static double g_localHitMs = 2.0;
static CachePolicy g_cachePolicy = CACHE_NONE;
static HiChainPaperParams g_params;

// ---------------------------------------------------------------------------
// Per-UAV cache (LRU order in cacheLru list front=MRU)
// ---------------------------------------------------------------------------
class NodeCacheState
{
  public:
    std::set<uint32_t> cachedBlocks;
    std::list<uint32_t> cacheLru;
    std::vector<uint32_t> cacheKeyOrder; // matches Python list(node.cache.keys()) for RL evict_slot
    std::set<uint32_t> encodedFragments;
    uint32_t nodeId = 0;

    void Touch(uint32_t blockId)
    {
        cacheLru.remove(blockId);
        cacheLru.push_front(blockId);
    }

    void InsertCache(uint32_t blockId)
    {
        if (cachedBlocks.count(blockId))
        {
            Touch(blockId);
            return;
        }
        cachedBlocks.insert(blockId);
        cacheKeyOrder.push_back(blockId);
        Touch(blockId);
    }

    void Evict(uint32_t blockId)
    {
        cachedBlocks.erase(blockId);
        cacheLru.remove(blockId);
        cacheKeyOrder.erase(
            std::remove(cacheKeyOrder.begin(), cacheKeyOrder.end(), blockId),
            cacheKeyOrder.end());
    }

    size_t CacheBytes() const
    {
        return cachedBlocks.size() * g_blockSizeBytes;
    }
};

// ---------------------------------------------------------------------------
// Block access application (UDP responder + PerformAccess)
// ---------------------------------------------------------------------------
class HiChainBlockAccessApp : public Application
{
  public:
    static TypeId GetTypeId();
    void SetNodeId(uint32_t id);
    void SetTimeouts(Time remote, Time recovery);

    bool PerformAccess(uint32_t blockId, uint32_t reqSeq, Time remoteTimeout, Time recoveryTimeout);
    void ApplyReplacement(uint32_t evictSlot, uint32_t addBlockId);

  private:
    void StartApplication() override;
    void StopApplication() override;
    void SendPacket(HiChainPacket pkt, Ipv4Address dest);
    void HandleRead(Ptr<Socket> socket);
    void CompleteAccess(uint32_t blockId, uint32_t reqSeq, HitType hit, Time start);
    void HandleRemoteTimeout(uint32_t blockId, uint32_t reqSeq, Time start);
    void HandleRecoveryTimeout(uint32_t blockId, uint32_t reqSeq, Time start);

    Ptr<Socket> m_socket;
    uint32_t m_nodeId = 0;
    std::shared_ptr<NodeCacheState> m_state;
    Time m_remoteTimeout;
    Time m_recoveryTimeout;

    struct PendingRequest
    {
        uint32_t blockId;
        uint32_t reqSeq;
        Time start;
        bool answered;
        uint32_t fragmentCount;
    };
    std::map<uint32_t, PendingRequest> m_pending;
};

NS_OBJECT_ENSURE_REGISTERED(HiChainBlockAccessApp);

TypeId
HiChainBlockAccessApp::GetTypeId()
{
    static TypeId tid = TypeId("ns3::HiChainBlockAccessApp")
                            .SetParent<Application>()
                            .AddConstructor<HiChainBlockAccessApp>();
    return tid;
}

void
HiChainBlockAccessApp::SetNodeId(uint32_t id)
{
    m_nodeId = id;
}

void
HiChainBlockAccessApp::SetTimeouts(Time remote, Time recovery)
{
    m_remoteTimeout = remote;
    m_recoveryTimeout = recovery;
}

void
HiChainBlockAccessApp::StartApplication()
{
    m_state = g_nodeState[m_nodeId];
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    m_socket = Socket::CreateSocket(GetNode(), tid);
    m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), g_port));
    m_socket->SetRecvCallback(MakeCallback(&HiChainBlockAccessApp::HandleRead, this));
    m_socket->SetAllowBroadcast(true);
}

void
HiChainBlockAccessApp::StopApplication()
{
    if (m_socket)
    {
        m_socket->Close();
    }
}

bool
HiChainBlockAccessApp::PerformAccess(uint32_t blockId, uint32_t reqSeq, Time remoteTimeout,
                                     Time recoveryTimeout)
{
    Time start = Simulator::Now();
    if (m_state->cachedBlocks.count(blockId))
    {
        m_state->Touch(blockId);
        double ms = g_localHitMs + (blockId % 7) * 0.1;
        g_records.push_back({m_nodeId, blockId, ms, HIT_LOCAL});
        return true;
    }

    PendingRequest pr{blockId, reqSeq, start, false, 0};
    m_pending[reqSeq] = pr;

    HiChainPacket pkt;
    pkt.type = PKT_BLOCK_REQUEST;
    pkt.blockId = blockId;
    pkt.srcNodeId = m_nodeId;
    pkt.reqSeq = reqSeq;
    SendPacket(pkt, Ipv4Address("255.255.255.255"));

    Simulator::Schedule(remoteTimeout, &HiChainBlockAccessApp::HandleRemoteTimeout, this, blockId,
                        reqSeq, start);
    return false;
}

void
HiChainBlockAccessApp::ApplyReplacement(uint32_t evictSlot, uint32_t addBlockId)
{
    if (addBlockId >= g_numBlocks)
    {
        return;
    }
    if (m_state->CacheBytes() < g_cacheCapacityBytes * g_cacheHighWatermark)
    {
        return;
    }
    if (m_state->cacheLru.empty())
    {
        return;
    }
    if (m_state->cacheKeyOrder.empty())
    {
        return;
    }
    uint32_t victim = m_state->cacheKeyOrder[evictSlot % m_state->cacheKeyOrder.size()];
    m_state->Evict(victim);
    m_state->InsertCache(addBlockId);
}

void
HiChainBlockAccessApp::CompleteAccess(uint32_t blockId, uint32_t reqSeq, HitType hit, Time start)
{
    auto it = m_pending.find(reqSeq);
    if ( it == m_pending.end())
    {
        return;
    }
    double ms = (Simulator::Now() - start).GetMilliSeconds();
    g_records.push_back({m_nodeId, blockId, ms, hit});
    m_pending.erase(it);
    if (hit == HIT_REMOTE || hit == HIT_RECOVERY)
    {
        m_state->InsertCache(blockId);
    }
}

void
HiChainBlockAccessApp::SendPacket(HiChainPacket pkt, Ipv4Address dest)
{
    Ptr<Packet> p = Create<Packet>(reinterpret_cast<uint8_t*>(&pkt), sizeof(pkt));
    m_socket->SendTo(p, 0, InetSocketAddress(dest, g_port));
}

void
HiChainBlockAccessApp::HandleRead(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from)))
    {
        if (packet->GetSize() < sizeof(HiChainPacket))
        {
            continue;
        }
        HiChainPacket pkt;
        packet->CopyData(reinterpret_cast<uint8_t*>(&pkt), sizeof(pkt));
        if (pkt.srcNodeId == m_nodeId)
        {
            continue;
        }

        if (pkt.type == PKT_BLOCK_REQUEST)
        {
            if (m_state->cachedBlocks.count(pkt.blockId))
            {
                HiChainPacket rsp{PKT_BLOCK_RESPONSE, pkt.blockId, m_nodeId, pkt.reqSeq};
                SendPacket(rsp, InetSocketAddress(Ipv4Address::ConvertFrom(from).GetIpv4(), g_port));
            }
            else if (m_state->encodedFragments.count(pkt.blockId))
            {
                HiChainPacket frag{PKT_FRAGMENT_OFFER, pkt.blockId, m_nodeId, pkt.reqSeq};
                SendPacket(frag, InetSocketAddress(Ipv4Address::ConvertFrom(from).GetIpv4(), g_port));
            }
        }
        else if (pkt.type == PKT_BLOCK_RESPONSE)
        {
            auto it = m_pending.find(pkt.reqSeq);
            if (it != m_pending.end() && it->second.blockId == pkt.blockId && !it->second.answered)
            {
                it->second.answered = true;
                CompleteAccess(pkt.blockId, pkt.reqSeq, HIT_REMOTE, it->second.start);
            }
        }
        else if (pkt.type == PKT_FRAGMENT_OFFER)
        {
            auto it = m_pending.find(pkt.reqSeq);
            if (it != m_pending.end() && it->second.blockId == pkt.blockId && !it->second.answered)
            {
                it->second.fragmentCount++;
                uint32_t need = g_kRecoverGlobal;
                auto bm = g_blockMeta.find(pkt.blockId);
                if (bm != g_blockMeta.end())
                {
                    need = bm->second.kRecover;
                }
                if (it->second.fragmentCount >= need)
                {
                    it->second.answered = true;
                    CompleteAccess(pkt.blockId, pkt.reqSeq, HIT_RECOVERY, it->second.start);
                }
            }
        }
    }
}

void
HiChainBlockAccessApp::HandleRemoteTimeout(uint32_t blockId, uint32_t reqSeq, Time start)
{
    auto it = m_pending.find(reqSeq);
    if (it == m_pending.end() || it->second.answered)
    {
        return;
    }
    Simulator::Schedule(m_recoveryTimeout - m_remoteTimeout, &HiChainBlockAccessApp::HandleRecoveryTimeout,
                        this, blockId, reqSeq, start);
}

void
HiChainBlockAccessApp::HandleRecoveryTimeout(uint32_t blockId, uint32_t reqSeq, Time start)
{
    auto it = m_pending.find(reqSeq);
    if (it == m_pending.end() || it->second.answered)
    {
        return;
    }
    double ms = (Simulator::Now() - start).GetMilliSeconds();
    g_records.push_back({m_nodeId, blockId, ms, HIT_MISS});
    m_pending.erase(it);
}

// ---------------------------------------------------------------------------
// Coordinator: one access per step (matches CacheReplacementEnv)
// ---------------------------------------------------------------------------
class HiChainCoordinatorApp : public Application
{
  public:
    static TypeId GetTypeId();
    void Setup(uint32_t nNodes, Time interval, Time remoteTimeout, Time recoveryTimeout);

  private:
    void StartApplication() override;
    void ScheduleStep();
    void PickRlAction(uint32_t step, uint32_t& evictSlot, uint32_t& addBlock);
    void PickLruAction(uint32_t nodeId, uint32_t reqBlock, uint32_t& evictSlot, uint32_t& addBlock);
    void PickFreqAction(uint32_t nodeId, uint32_t& evictSlot, uint32_t& addBlock);

    uint32_t m_nNodes = 0;
    uint32_t m_step = 0;
    uint32_t m_reqSeq = 0;
    Time m_interval;
    Time m_remoteTimeout;
    Time m_recoveryTimeout;
    std::vector<Ptr<HiChainBlockAccessApp>> m_apps;
};

NS_OBJECT_ENSURE_REGISTERED(HiChainCoordinatorApp);

TypeId
HiChainCoordinatorApp::GetTypeId()
{
    static TypeId tid = TypeId("ns3::HiChainCoordinatorApp")
                            .SetParent<Application>()
                            .AddConstructor<HiChainCoordinatorApp>();
    return tid;
}

void
HiChainCoordinatorApp::Setup(uint32_t nNodes, Time interval, Time remoteTimeout, Time recoveryTimeout)
{
    m_nNodes = nNodes;
    m_interval = interval;
    m_remoteTimeout = remoteTimeout;
    m_recoveryTimeout = recoveryTimeout;
}

void
HiChainCoordinatorApp::StartApplication()
{
    ScheduleStep();
}

void
HiChainCoordinatorApp::ScheduleStep()
{
    if (Simulator::Now() >= GetStopTime())
    {
        return;
    }
    if (m_step >= g_accessTrace.size())
    {
        return;
    }

    uint32_t blockId = g_accessTrace[m_step];
    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    uint32_t nodeId = uv->GetInteger(0, m_nNodes - 1);

    if (nodeId >= g_accessApps.size() || !g_accessApps[nodeId])
    {
        m_step++;
        Simulator::Schedule(m_interval, &HiChainCoordinatorApp::ScheduleStep, this);
        return;
    }

    uint32_t reqSeq = ++m_reqSeq;
    bool hit = g_accessApps[nodeId]->PerformAccess(blockId, reqSeq, m_remoteTimeout, m_recoveryTimeout);

    if (!hit && g_cachePolicy != CACHE_NONE)
    {
        uint32_t evictSlot = 0, addBlock = blockId;
        if (g_cachePolicy == CACHE_RL)
        {
            PickRlAction(m_step, evictSlot, addBlock);
        }
        else if (g_cachePolicy == CACHE_LRU)
        {
            PickLruAction(nodeId, blockId, evictSlot, addBlock);
        }
        else if (g_cachePolicy == CACHE_FREQ)
        {
            PickFreqAction(nodeId, evictSlot, addBlock);
        }
        for (uint32_t i = 0; i < m_nNodes; ++i)
        {
            g_accessApps[i]->ApplyReplacement(evictSlot, addBlock);
        }
    }

    m_step++;
    Simulator::Schedule(m_interval, &HiChainCoordinatorApp::ScheduleStep, this);
}

void
HiChainCoordinatorApp::PickRlAction(uint32_t step, uint32_t& evictSlot, uint32_t& addBlock)
{
    for (const auto& row : g_rlActions)
    {
        if (row.step == step)
        {
            evictSlot = row.evictSlot;
            addBlock = row.addBlockId;
            return;
        }
    }
}

void
HiChainCoordinatorApp::PickLruAction(uint32_t nodeId, uint32_t reqBlock, uint32_t& evictSlot,
                                     uint32_t& addBlock)
{
    addBlock = reqBlock;
    evictSlot = 0;
    auto st = g_nodeState[nodeId];
    if (st->cacheLru.empty())
    {
        return;
    }
    uint32_t victim = st->cacheLru.back();
    for (size_t i = 0; i < st->cacheKeyOrder.size(); ++i)
    {
        if (st->cacheKeyOrder[i] == victim)
        {
            evictSlot = static_cast<uint32_t>(i);
            break;
        }
    }
}

static uint32_t
BlockIdForHighestFreqNotCached(uint32_t requestingNode)
{
    uint32_t best = 0;
    double bestF = -1;
    auto st = g_nodeState[requestingNode];
    for (const auto& kv : g_blockMeta)
    {
        if (st->cachedBlocks.count(kv.first))
        {
            continue;
        }
        if (kv.second.frequency > bestF)
        {
            bestF = kv.second.frequency;
            best = kv.first;
        }
    }
    return best;
}

void
HiChainCoordinatorApp::PickFreqAction(uint32_t nodeId, uint32_t& evictSlot, uint32_t& addBlock)
{
    auto st = g_nodeState[nodeId];
    double minFreq = 1e18;
    size_t idx = 0;
    size_t pickIdx = 0;
    for (uint32_t bid : st->cacheKeyOrder)
    {
        double f = 1.0;
        auto it = g_blockMeta.find(bid);
        if (it != g_blockMeta.end())
        {
            f = it->second.frequency;
        }
        if (f < minFreq)
        {
            minFreq = f;
            pickIdx = idx;
        }
        idx++;
    }
    evictSlot = static_cast<uint32_t>(pickIdx);
    addBlock = BlockIdForHighestFreqNotCached(nodeId);
}

static bool
LoadAccessTrace(const std::string& path)
{
    std::ifstream in(path);
    if (!in.good())
    {
        return false;
    }
    std::string header;
    std::getline(in, header);
    uint32_t step, blockId;
    char comma;
    while (in >> step >> comma >> blockId)
    {
        g_accessTrace.push_back(blockId);
    }
    return !g_accessTrace.empty();
}

static bool
LoadRlActions(const std::string& path)
{
    std::ifstream in(path);
    if (!in.good())
    {
        return false;
    }
    std::string header;
    std::getline(in, header);
    RlActionRow row;
    char comma;
    while (in >> row.step >> comma >> row.evictSlot >> comma >> row.addBlockId)
    {
        g_rlActions.push_back(row);
    }
    return !g_rlActions.empty();
}

static bool
LoadBlockMetaCsv(const std::string& path)
{
    std::ifstream in(path);
    if (!in.good())
    {
        return false;
    }
    std::string header;
    std::getline(in, header);
    uint32_t bid;
    double freq;
    uint32_t k;
    char comma;
    while (in >> bid >> comma >> freq >> comma >> k)
    {
        g_blockMeta[bid] = {freq, k};
    }
    return !g_blockMeta.empty();
}

static void
WriteMetricsCsv(const std::string& path, uint32_t nUavs, double simSeconds,
                const std::string& cachePolicyName)
{
    uint32_t local = 0, remote = 0, recovery = 0, miss = 0;
    double sumLat = 0;
    for (const auto& r : g_records)
    {
        sumLat += r.latencyMs;
        switch (r.hitType)
        {
        case HIT_LOCAL:
            local++;
            break;
        case HIT_REMOTE:
            remote++;
            break;
        case HIT_RECOVERY:
            recovery++;
            break;
        default:
            miss++;
            break;
        }
    }
    uint32_t total = local + remote + recovery + miss;
    double hitRate = total ? (local + remote) * 1.0 / total : 0;
    double availRate = total ? (local + remote + recovery) * 1.0 / total : 0;
    double avgLat = total ? sumLat / total : 0;

    bool exists = std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!exists)
    {
        out << "n_uavs,sim_time_s,cache_policy,num_accesses,cache_hit_rate,data_availability_rate,"
               "avg_access_latency_ms,local,remote,recovery,miss\n";
    }
    out << nUavs << "," << simSeconds << "," << cachePolicyName << "," << total << "," << hitRate
        << "," << availRate << "," << avgLat << "," << local << "," << remote << "," << recovery
        << "," << miss << "\n";
    std::cout << "policy=" << cachePolicyName << " n_uavs=" << nUavs << " hit_rate=" << hitRate
              << " avg_ms=" << avgLat << " -> " << path << std::endl;
}

static std::string
CachePolicyName(CachePolicy p)
{
    switch (p)
    {
    case CACHE_LRU:
        return "lru";
    case CACHE_FREQ:
        return "freq";
    case CACHE_RL:
        return "rl";
    default:
        return "none";
    }
}

int
main(int argc, char* argv[])
{
    HiChainPaperParams paper;
    uint32_t nUavs = paper.nUavsSwarm;
    uint32_t numBlocks = paper.numBlocks;
    uint32_t blockSizeMb = paper.blockSizeMb;
    uint32_t cacheMb = paper.cachePerNodeMb;
    uint32_t storageMb = paper.storagePerNodeMb;
    uint32_t kRecover = paper.kRecover;
    double simTime = paper.simTimeS;
    double accessIntervalMs = paper.accessIntervalMs;
    double areaM = paper.areaSideM;
    std::string csvOut = "hi-chain-swarm-metrics.csv";
    std::string blockCsv;
    std::string accessTracePath;
    std::string rlActionsPath;
    std::string cachePolicyStr = "none";
    bool enablePcap = false;

    CommandLine cmd;
    cmd.AddValue("nUavs", "Number of UAV nodes", nUavs);
    cmd.AddValue("numBlocks", "Number of blocks", numBlocks);
    cmd.AddValue("kRecover", "Default RS fragments for recovery", kRecover);
    cmd.AddValue("blockSizeMb", "Block size MB", blockSizeMb);
    cmd.AddValue("cachePerNodeMb", "Cache per UAV MB", cacheMb);
    cmd.AddValue("storagePerNodeMb", "Encoded storage per UAV MB", storageMb);
    cmd.AddValue("simTime", "Simulation time s", simTime);
    cmd.AddValue("accessIntervalMs", "Access interval ms", accessIntervalMs);
    cmd.AddValue("areaM", "Deployment area m", areaM);
    cmd.AddValue("blockCsv", "block_frequencies.csv path (optional)", blockCsv);
    cmd.AddValue("accessTrace", "ns3_access_trace.csv from Python", accessTracePath);
    cmd.AddValue("rlActions", "ns3_rl_actions.csv from export_ns3_rl_trace.py", rlActionsPath);
    cmd.AddValue("cachePolicy", "none|lru|freq|rl", cachePolicyStr);
    cmd.AddValue("csvOut", "Output CSV", csvOut);
    cmd.AddValue("enablePcap", "PCAP", enablePcap);
    cmd.Parse(argc, argv);

    g_params = paper;
    g_numBlocks = numBlocks;
    g_blockSizeBytes = blockSizeMb * 1024U * 1024U;
    g_cacheCapacityBytes = cacheMb * 1024U * 1024U;
    g_kRecoverGlobal = kRecover;
    g_localHitMs = paper.localHitProcessingMs;
    g_cacheHighWatermark = paper.cacheHighWatermark;

    if (cachePolicyStr == "lru")
    {
        g_cachePolicy = CACHE_LRU;
    }
    else if (cachePolicyStr == "freq")
    {
        g_cachePolicy = CACHE_FREQ;
    }
    else if (cachePolicyStr == "rl")
    {
        g_cachePolicy = CACHE_RL;
    }
    else
    {
        g_cachePolicy = CACHE_NONE;
    }

    g_records.clear();
    g_nodeState.clear();
    g_accessTrace.clear();
    g_rlActions.clear();
    g_blockMeta.clear();
    g_accessApps.clear();

    if (!blockCsv.empty())
    {
        LoadBlockMetaCsv(blockCsv);
    }
    if (!accessTracePath.empty())
    {
        if (!LoadAccessTrace(accessTracePath))
        {
            NS_LOG_ERROR("Failed to load access trace");
        }
    }
    if (g_cachePolicy == CACHE_RL)
    {
        if (!LoadRlActions(rlActionsPath))
        {
            std::cerr << "cachePolicy=rl requires --rlActions CSV from export_ns3_rl_trace.py\n";
            return 1;
        }
    }

    if (g_accessTrace.empty())
    {
        for (uint32_t i = 0; i < numBlocks && g_accessTrace.size() < 100000; ++i)
        {
            g_accessTrace.push_back(i % numBlocks);
        }
    }

    NodeContainer nodes;
    nodes.Create(nUavs);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode",
                                 StringValue("OfdmRate6Mbps"), "ControlMode",
                                 StringValue("OfdmRate6Mbps"));
    YansWifiPhyHelper phy;
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    phy.SetChannel(channel.Create());
    phy.Set("TxPowerStart", DoubleValue(20));
    phy.Set("TxPowerEnd", DoubleValue(20));
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    address.Assign(devices);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
    uint32_t grid = static_cast<uint32_t>(std::ceil(std::sqrt(nUavs)));
    for (uint32_t i = 0; i < nUavs; ++i)
    {
        double x = (i % grid) * (areaM / std::max(1u, grid - 1));
        double y = (i / grid) * (areaM / std::max(1u, grid - 1));
        posAlloc->Add(Vector(x, y, paper.uavAltitudeM));
    }
    mobility.SetPositionAllocator(posAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    uint32_t cacheSlots = std::max(5u, numBlocks / (nUavs * 4));
    for (uint32_t i = 0; i < nUavs; ++i)
    {
        auto st = std::make_shared<NodeCacheState>();
        st->nodeId = i;
        for (uint32_t c = 0; c < cacheSlots; ++c)
        {
            uint32_t bid = uv->GetInteger(0, numBlocks - 1);
            st->InsertCache(bid);
        }
        for (uint32_t b = 0; b < numBlocks; ++b)
        {
            if (uv->GetValue(0, 1) < 0.7)
            {
                st->encodedFragments.insert(b);
            }
        }
        g_nodeState[i] = st;
    }

    Time remoteT = MilliSeconds(paper.remoteTimeoutMs);
    Time recoveryT = MilliSeconds(paper.recoveryTimeoutMs);
    g_accessApps.resize(nUavs);
    for (uint32_t i = 0; i < nUavs; ++i)
    {
        Ptr<HiChainBlockAccessApp> app = CreateObject<HiChainBlockAccessApp>();
        app->SetNodeId(i);
        app->SetTimeouts(remoteT, recoveryT);
        nodes.Get(i)->AddApplication(app);
        app->SetStartTime(Seconds(1.0));
        app->SetStopTime(Seconds(simTime));
        g_accessApps[i] = app;
    }

    Ptr<HiChainCoordinatorApp> coord = CreateObject<HiChainCoordinatorApp>();
    coord->Setup(nUavs, MilliSeconds(accessIntervalMs), remoteT, recoveryT);
    nodes.Get(0)->AddApplication(coord);
    coord->SetStartTime(Seconds(2.0));
    coord->SetStopTime(Seconds(simTime));

    if (enablePcap)
    {
        phy.EnablePcapAll("hi-chain-uav");
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();

    WriteMetricsCsv(csvOut, nUavs, simTime, CachePolicyName(g_cachePolicy));
    return 0;
}
