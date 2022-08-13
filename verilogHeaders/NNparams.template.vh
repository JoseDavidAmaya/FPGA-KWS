localparam Qm = 3;
localparam Qn = 4;

localparam memwidth={MEMWIDTH};
localparam elemwidth=8;
localparam m=memwidth/elemwidth;

// ----- Parameters
localparam memParamsInitFile = "{MEM_PARAMS_INIT_FILE}";
localparam memInputInitFileSim = "{MEM_INPUT_INIT_FILE_SIM}";
localparam memOutputInitFileSim = "{MEM_OUTPUT_INIT_FILE_SIM}";

localparam mem7segLabels = "{MEM_7SEG_LABELS}"
localparam mem7segEncoder = "{MEM_7SEG_ENCODER}"


localparam gru_out_size = {GRU_OUT_SIZE}; // Size of the output of the GRU layer [Also the size (or a factor of the size) of the output of its operations] (in memory)

/// GRU kernel
localparam memdepth_gk = {MEMDEPTH_GRU_KERNEL};
localparam memdepth_gk_individual = memdepth_gk/3; // GRU kernel is the concatenation of 3 matrices, this is the depth of each individual matrix
localparam addr_st_gk = 0;

/// GRU recurrent kernel
localparam memdepth_grk = {MEMDEPTH_GRU_REC_KERNEL};
localparam memdepth_grk_individual = memdepth_grk/3; // GRU recurrent kernel is the concatenation of 3 matrices, this is the depth of each individual matrix
localparam addr_st_grk = addr_st_gk + memdepth_gk;

/// GRU bias
localparam memdepth_gb = {MEMDEPTH_GRU_BIAS};
localparam addr_st_gb = addr_st_grk + memdepth_grk;

/// FC kernel
localparam memdepth_fk = {MEMDEPTH_FC_KERNEL};
localparam addr_st_fk = addr_st_gb + memdepth_gb;

/// FC bias
localparam memdepth_fb = {MEMDEPTH_FC_BIAS};
localparam addr_st_fb = addr_st_fk + memdepth_fk;

localparam memdepthParams = memdepth_gk + memdepth_grk + memdepth_gb + memdepth_fk + memdepth_fb; // {MEMDEPTH_PARAMS}
localparam addrwidth_params = $clog2(memdepthParams);
        
// ----- Input data
localparam feature_size = {INPUT_FEATURE_SIZE}; // Size of each timestep of the input (in memory)
localparam num_timesteps = {NUM_INPUT_TIMESTEPS}; // Number of timesteps in the input (in memory)
localparam memdepthInData = feature_size*num_timesteps;
localparam addrwidth_InData = $clog2(memdepthInData);

// ----- Temporary data / Output data 1
localparam memdepthTempData1 = 5*gru_out_size;
localparam addrwidth_TempData1 = $clog2(memdepthTempData1);

// ----- Temporary data / Output data 1
localparam memdepthTempData2 = 2*gru_out_size;
localparam addrwidth_TempData2 = $clog2(memdepthTempData2);

// ---- argmax block config
localparam argmax_vec_st = 0;
localparam argmax_vec_end = {ARGMAX_VECEND}; // 2
localparam argmax_last_vec_size = {ARGMAX_LAST_VEC_SIZE}; // 4
