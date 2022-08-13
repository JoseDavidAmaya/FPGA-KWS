`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/20/2022 08:57:11 AM
// Design Name: 
// Module Name: test2
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module test2(
        input clk,
        input rst,
        input st,
        
        input [2:0]mem_select_in1,
        input [2:0]mem_select_in2,
        input [2:0]mem_select_out,
        
        input [addr_width_max-1:0]mem_addr_in2, // addr_width_max = 17
        output [memwidth-1:0]mem_r_data_in2, // memwidth = 64
        input mem_en_in2,
        
        output wand rdy
    );
    genvar i;
    
    localparam memwidth=64;
    localparam memoryReadLatency = 2;   
    
    // ----- Parameters
    
    /// GRU kernel
    localparam memdepth_gk = 6000;
    localparam addr_st_gk = 0;
    
    /// GRU recurrent kernel
    localparam memdepth_grk = 60000;
    localparam addr_st_grk = addr_st_gk + memdepth_gk;
    
    /// GRU bias
    localparam memdepth_gb = 150;
    localparam addr_st_gb = addr_st_grk + memdepth_grk;
    
    /// FC kernel
    localparam memdepth_fk = 800;
    localparam addr_st_fk = addr_st_gb + memdepth_gb;
    
    /// FC bias
    localparam memdepth_fb = 2;
    localparam addr_st_fb = addr_st_fk + memdepth_fk;
    
    localparam memdepthParams = memdepth_gk + memdepth_grk + memdepth_gb + memdepth_fk + memdepth_fb;
    localparam addrwidth_params = $clog2(memdepthParams);
        
    // ----- Input data
    localparam memdepthInData = 250;
    localparam addrwidth_InData = $clog2(memdepthInData);
    
    // ----- Temporary data / Output data 1
    localparam memdepthTempData1 = 250;
    localparam addrwidth_TempData1 = $clog2(memdepthTempData1);
    
    // ----- Temporary data / Output data 1
    localparam memdepthTempData2 = 100;
    localparam addrwidth_TempData2 = $clog2(memdepthTempData2);
    
    // ----- Memories

    /// params
    wire [memwidth-1:0]r_data_params;
    wor [addrwidth_params - 1:0]addr_params;
    wor r_en_params;
    
    /// indata
    wire [memwidth - 1:0]r_data_indata;
    wor [addrwidth_InData - 1:0]addr_indata;
    reg [addrwidth_InData - 1:0]addr_indata_offset = 0;
    wor [memwidth - 1:0]w_data_indata;
    wor en_indata;
    wor w_en_indata;

    /// tempdata1
    //// Port A
    wire [memwidth - 1:0]r_data_tempA;
    wor [addrwidth_TempData1 - 1:0]addr_tempA;
    wor [memwidth - 1:0]w_data_tempA;
    wor en_tempA;
    wor w_en_tempA;
    
    //// Port B
    wire [memwidth - 1:0]r_data_tempB;
    wor [addrwidth_TempData1 - 1:0]addr_tempB;
    wor [memwidth - 1:0]w_data_tempB;
    wor en_tempB;
    wor w_en_tempB;
    
    /// tempdata2
    //// port C
    wire [memwidth - 1:0]r_data_tempC;
    wor [addrwidth_TempData1 - 1:0]addr_tempC;
    wor [memwidth - 1:0]w_data_tempC;
    wor en_tempC;
    wor w_en_tempC;
    
    memories
    #(
        .memwidth(memwidth),
        .memoryReadLatency(memoryReadLatency),
        
        .memdepthParams(memdepthParams),
        .memdepthInData(memdepthInData),
        .memdepthTempData1(memdepthTempData1),
        .memdepthTempData2(memdepthTempData2),
        
        .parametersInitFile("C:/Users/jdah1/Desktop/ASR/GRU.mem")
    )
    
    mem(
        .clk(clk),
        .rst(rst),
        
        .r_data_params(r_data_params),
        .addr_params(addr_params),
        .r_en_params(r_en_params),
        
        .r_data_indata(r_data_indata),
        .addr_indata(addr_indata + addr_indata_offset),
        .w_data_indata(w_data_indata),
        .en_indata(en_indata),
        .w_en_indata(w_en_indata),
        
        .r_data_tempA(r_data_tempA),
        .addr_tempA(addr_tempA),
        .w_data_tempA(w_data_tempA),
        .en_tempA(en_tempA),
        .w_en_tempA(w_en_tempA),
        
        .r_data_tempB(r_data_tempB),
        .addr_tempB(addr_tempB),
        .w_data_tempB(w_data_tempB),
        .en_tempB(en_tempB),
        .w_en_tempB(w_en_tempB),
        
        .r_data_tempC(r_data_tempC),
        .addr_tempC(addr_tempC),
        .w_data_tempC(w_data_tempC),
        .en_tempC(en_tempC),
        .w_en_tempC(w_en_tempC)
    );

    
    /// Mem select
    localparam MEMPARAMS = 3'd0;
    localparam MEMIN = 3'd1;
    localparam MEMTEMPA = 3'd2;
    localparam MEMTEMPB = 3'd3;
    localparam MEMTEMPC = 3'd4;
    
    localparam IN1 = 3'd0;
    localparam IN2 = 3'd1;
    localparam OUT = 3'd2;
    
    localparam addr_width_max = addrwidth_params;

    reg [2:0]mem_select[2:0]; // 0: in1, 1: in2, 2: out
    wire [7:0]onehot_mem_select[2:0];
    
    for (i = 0; i < 3; i=i+1) begin
        assign onehot_mem_select[i] = 8'b1 << mem_select[i];
    end
    
    always @* begin
        mem_select[IN1] <= mem_select_in1; // in1
        mem_select[IN2] <= mem_select_in2; // in2
        mem_select[OUT] <= mem_select_out; // out
    end

    wor [addr_width_max-1:0]mem_addr[2:0]; // 0: in1, 1: in2, 2: out
    wor [memwidth-1:0]mem_data[2:0];
    wor mem_en[2:0];
    
    // ---- NOTE: This assignments allows reading from IN2, so we may select any memory as IN2 and read it
    assign mem_addr[IN2] = mem_addr_in2;
    assign mem_r_data_in2 = mem_data[IN2];
    assign mem_en[IN2] = mem_en_in2;
    // ----
    
    for (i = 0; i < 3; i=i+1) begin
        // Addr 'demux'
        assign addr_params = onehot_mem_select[i][MEMPARAMS] ? mem_addr[i] : 0;
        assign addr_indata = onehot_mem_select[i][MEMIN] ? mem_addr[i] : 0;
        assign addr_tempA = onehot_mem_select[i][MEMTEMPA] ? mem_addr[i] : 0;
        assign addr_tempB = onehot_mem_select[i][MEMTEMPB] ? mem_addr[i] : 0;
        assign addr_tempC = onehot_mem_select[i][MEMTEMPC] ? mem_addr[i] : 0;
        
        // En 'demux'
        assign r_en_params = onehot_mem_select[i][MEMPARAMS] ? mem_en[i] : 0;
        assign en_indata = onehot_mem_select[i][MEMIN] ? mem_en[i] : 0;
        assign en_tempA = onehot_mem_select[i][MEMTEMPA] ? mem_en[i] : 0;
        assign en_tempB = onehot_mem_select[i][MEMTEMPB] ? mem_en[i] : 0;
        assign en_tempC = onehot_mem_select[i][MEMTEMPC] ? mem_en[i] : 0;
        if (i == 2) begin
            assign w_en_indata = onehot_mem_select[i][MEMIN] ? mem_en[i] : 0;
            assign w_en_tempA = onehot_mem_select[i][MEMTEMPA] ? mem_en[i] : 0;
            assign w_en_tempB = onehot_mem_select[i][MEMTEMPB] ? mem_en[i] : 0;
            assign w_en_tempC = onehot_mem_select[i][MEMTEMPC] ? mem_en[i] : 0;
        end
        
        // r_data 'mux'
        if (i < 2) begin
            assign mem_data[i] = onehot_mem_select[i][MEMPARAMS] ? r_data_params : 0;
            assign mem_data[i] = onehot_mem_select[i][MEMIN] ? r_data_indata : 0;
            assign mem_data[i] = onehot_mem_select[i][MEMTEMPA] ? r_data_tempA : 0;
            assign mem_data[i] = onehot_mem_select[i][MEMTEMPB] ? r_data_tempB : 0;
            assign mem_data[i] = onehot_mem_select[i][MEMTEMPC] ? r_data_tempC : 0;
        end
            
        // w_data 'demux'
        if (i == 2) begin
            assign w_data_indata = onehot_mem_select[i][MEMIN] ? mem_data[i] : 0;
            assign w_data_tempA = onehot_mem_select[i][MEMTEMPA] ? mem_data[i] : 0;
            assign w_data_tempB = onehot_mem_select[i][MEMTEMPB] ? mem_data[i] : 0;
            assign w_data_tempC = onehot_mem_select[i][MEMTEMPC] ? mem_data[i] : 0;
        end
    end
    
    // ----- Operations
    
    /// Operation select

    //// st
    localparam VECMATMULT = 3'd0;
    localparam VECVECFUNC = 3'd1;
    localparam VECFUNC = 3'd2;
    
    reg [2:0]op_select = 0;
    wire [5:0]onehot_op_select = st << op_select;
    
    //// vecVecFunc func select
    localparam MULT = 1'b0;
    localparam ADD = 1'b1;
    reg vecVecFunc_select = 0;
    
    //// vecFunc func select
    localparam SIGM = 2'd0;
    localparam TANH = 2'd1;
    localparam ONEMA = 2'd2;
    reg [1:0]vecFunc_select = 0;

    //// Addresses
    reg [addr_width_max-1:0]addr_st[2:0];
    reg [addr_width_max-1:0]addr_end[2:0];
    reg [addr_width_max-1:0]num_positions;
    
    //// Block config
    
    ///// vecMatMult
    localparam addr_width_maxvec = addrwidth_TempData1;
    reg [4:0]vmm_addr_vecin_st_offset = 0;

    ///// vecVecFunc
    reg vvf_disable_addr_vecout = 0;

    ///// vecFunc
    reg [4:0]vf_addr_vecin_st_offset = 0;
    
    ////// NOTE: TESTING vecMatMult
    
    
    initial begin
        op_select = VECMATMULT;
        
        addr_st[IN1] = addr_st_gk;
        addr_end[IN1] = addr_st_gk + memdepth_gk;
        
        addr_st[IN2] = 0;
        addr_end[IN2] = 5;
        
        addr_st[OUT] = 0;
        addr_end[OUT] = 150; // 150
        
        addr_indata_offset = 0;
        vmm_addr_vecin_st_offset = 0;
        
        /*
        addr_st[IN1] = addr_st_fk;
        addr_end[IN1] = addr_st_fk + memdepth_fk;
        
        addr_st[IN2] = 0;
        addr_end[IN2] = 50;
        
        addr_st[OUT] = 0;
        addr_end[OUT] = 2;
        */
    end
    
    
    
    ////// NOTE: TESTING vecVecFunc
    /*
    initial begin
        op_select = VECVECFUNC;
        vecVecFunc_select = MULT; // MULT ADD
        
        addr_st[IN1] = addr_st_gk;
        addr_st[IN2] = 0;
        addr_st[OUT] = 0;
        
        num_positions = 10;
        
        vvf_disable_addr_vecout = 1;
    end
    */
    
    
    ////// NOTE: TESTING vecFunc
    /*
    initial begin
        op_select = VECFUNC;
        vecFunc_select = SIGM; // SIGM TANH ONEMA
        
        addr_st[IN1] = addr_st_gk;
        addr_st[OUT] = 0;
        
        num_positions = 10;
        vf_addr_vecin_st_offset = 1;
    end
    */
    
    
    wor ren_inputs;
    assign mem_en[IN1] = ren_inputs;
    assign mem_en[IN2] = ren_inputs;
    
    /// Vector matrix multiplication
       
    vecMatMult
    #(
        .memwidth(memwidth),
        .elemwidth(8),
        
        .addr_width_mat(addrwidth_params),
        .addr_width_maxvec(addr_width_maxvec),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(3),
        .Qn(4)
    )
    
    vecmatmultblock(
        .clk(clk),
        .rst(rst),
        .st(onehot_op_select[VECMATMULT]),
        
        .r_data_mat(mem_data[IN1]),
        .r_data_vecin(mem_data[IN2]),
        
        .addr_mat_st(addr_st[IN1]),
        .addr_mat_end(addr_end[IN1]),
        .addr_vecin_st(addr_st[IN2]),
        .addr_vecin_end(addr_end[IN2]),
        .addr_vecin_st_offset(vmm_addr_vecin_st_offset),
        
        .addr_vecout_st(addr_st[OUT]),
        .addr_vecout_end(addr_end[OUT]),
        
        .rdy(rdy),
        
        .ren_inputs(ren_inputs),
        
        .addr_mat(mem_addr[IN1]),
        .addr_vecin(mem_addr[IN2]),
        
        .wen_vecout(mem_en[OUT]),
        
        .w_data_vecout(mem_data[OUT]),
        .addr_vecout(mem_addr[OUT])
    );
    
    /// Vector vector addition/multiplication
    vecVecFunc
    #(
        .memwidth(memwidth),
        .elemwidth(8),
        
        .addr_width_max(addr_width_maxvec),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(3),
        .Qn(4)
    )
    
    vecvecfuncblock (
        .clk(clk),
        .rst(rst),
        
        .st(onehot_op_select[VECVECFUNC]),
        
        .func_select(vecVecFunc_select),
        .disable_addr_vecout(vvf_disable_addr_vecout),
        
        .r_data_vecA(mem_data[IN1]),
        .r_data_vecB(mem_data[IN2]),
        
        .addr_vecA_st(addr_st[IN1]),
        .addr_vecB_st(addr_st[IN2]),
        
        .addr_vecout_st(addr_st[OUT]),
        
        .num_positions(num_positions),

        .rdy(rdy),
        
        .ren_inputs(ren_inputs),
        
        .addr_vecA(mem_addr[IN1]),
        .addr_vecB(mem_addr[IN2]),
        
        .wen_vecout(mem_en[OUT]),
        .w_data_vecout(mem_data[OUT]),
        .addr_vecout(mem_addr[OUT])
    );
    
    /// Vector element-wise sigm/tanh/oneMinusA
    vecFunc
    #(
        .memwidth(memwidth),
        .elemwidth(8),
        
        .addr_width_max(addr_width_maxvec),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(3),
        .Qn(4)
    )

    vecfuncblock (
        .clk(clk),
        .rst(rst),
        
        .st(onehot_op_select[VECFUNC]),
        
        .func_select(vecFunc_select),
        
        .r_data_vecin(mem_data[IN1]),
        
        .addr_vecin_st(addr_st[IN1]),
        .addr_vecin_st_offset(vf_addr_vecin_st_offset),
        
        .addr_vecout_st(addr_st[OUT]),
        
        .num_positions(num_positions),
        
        .rdy(rdy),
        
        .ren_inputs(ren_inputs),
        
        .addr_vecin(mem_addr[IN1]),
        
        .wen_vecout(mem_en[OUT]),
        .w_data_vecout(mem_data[OUT]),
        .addr_vecout(mem_addr[OUT])
    );
    
endmodule
