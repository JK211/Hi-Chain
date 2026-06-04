#ifndef HI_CHAIN_PAPER_PARAMS_H
#define HI_CHAIN_PAPER_PARAMS_H

/**
 * Default parameters aligned with Hi-Chain_Code (paper experiments).
 *
 * Sources:
 *   data_generator003.py __main__  -> 8000 blocks, n=50, zipf_s=0.4
 *   compare_of_storage_cost_fig6-9.py -> REF_NODE_COUNT=50, 1MB block, 10GB limit
 *   train_model007.py -> RL train often n=10, 1MB block, 20MB cache, 1GB storage
 *   CacheReplacementEnv -> delay_hit=1, delay_miss=10, cache replace at 90% full
 *
 * NS-3 swarm validation typically uses nUavs=50; RL policy must be trained/exported
 * with the SAME n_nodes (state dimension = 4 + 3*n).
 */
struct HiChainPaperParams
{
    // --- Dataset / encoding (data_generator003) ---
    uint32_t numBlocks = 8000;
    double zipfS = 0.4;
    uint32_t freqMin = 1;
    uint32_t freqMax = 1000;

    // --- Swarm scale (storage + NS-3 figures) ---
    uint32_t nUavsSwarm = 50;
    uint32_t nodesPerShard = 10;

    // --- RL training defaults (train_model007) ---
    uint32_t nUavsTrain = 10;

    // --- Block & storage ---
    uint32_t blockSizeMb = 1;
    uint32_t cachePerNodeMb = 20;
    uint32_t storagePerNodeMb = 1000;

    // --- RS recovery ---
    uint32_t kRecover = 20;

    // --- NS-3 simulation ---
    double simTimeS = 100.0;
    double accessIntervalMs = 50.0;
    double areaSideM = 500.0;
    double uavAltitudeM = 50.0;
    double remoteTimeoutMs = 120.0;
    double recoveryTimeoutMs = 250.0;
    double localHitProcessingMs = 2.0;

    // --- CacheReplacementEnv rewards / energy ---
    double delayHitMs = 1.0;
    double delayMissMs = 10.0;
    double energyHit = 0.1;
    double energyMiss = 1.0;
    double cacheHighWatermark = 0.9;

    // --- WiFi (adjust to match your NS-3.25 paper setup) ---
    const char* wifiStandard = "80211a";
    const char* wifiDataMode = "OfdmRate6Mbps";
    double txPowerDbm = 20.0;
};

#endif
