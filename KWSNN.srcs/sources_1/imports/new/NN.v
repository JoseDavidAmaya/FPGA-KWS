`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/26/2022 11:28:45 AM
// Design Name: 
// Module Name: NN
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

`ifndef SYNTHESIS
    `define simulation
    `define accessTempA
`endif

module NN(
        input clk,
        input rst,
        input st,
        
        // For simulation and testing
        `ifdef simulation
            output [memwidth - 1:0]access_r_data_temp,
            input [addrwidth_TempData1 - 1:0]access_addr_temp,
            input access_en_temp,
        `endif
        
        input [addrwidth_InData - 1:0]access_addr_indata,
        input [memwidth - 1:0]access_w_data_indata,
        input access_en_indata,
        input access_w_en_indata,
        
        
        output reg [7:0]outIndex,
        output rdy
    );
    genvar i;
    
    localparam memoryReadLatency = 2;
       
    `include "NNparams.vh"
    
    // ----- Memories

    /// params
    wire [memwidth-1:0]r_data_params;
    wor [addrwidth_params - 1:0]addr_params;
    wor r_en_params;
    
    /// indata
    wire [memwidth - 1:0]r_data_indata;
    wor [addrwidth_InData - 1:0]addr_indata;
    wor [memwidth - 1:0]w_data_indata;
    wor en_indata;
    wor w_en_indata;
    
    /// We can access indata from outside the module to write data to it while the module is idle
    
    assign addr_indata = rdy ? access_addr_indata : 0;
    assign w_data_indata = rdy ? access_w_data_indata : 0;
    assign en_indata = rdy ? access_en_indata : 0;
    assign w_en_indata = rdy ? access_w_en_indata : 0;
       
    /// --

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
    
    // -- For testing/simulation
    `ifdef simulation
        `ifdef accessTempA
            assign access_r_data_temp = rdy ? r_data_tempA : 0;
            assign addr_tempA = rdy ? access_addr_temp : 0;
            assign en_tempA = rdy ? access_en_temp : 0;
        `else
            assign access_r_data_temp = rdy ? r_data_tempC : 0;
            assign addr_tempC = rdy ? access_addr_temp : 0;
            assign en_tempC = rdy ? access_en_temp : 0;
        `endif
    `endif
    // --
    
    memories
    #(
        .memwidth(memwidth),
        .memoryReadLatency(memoryReadLatency),
        
        .memdepthParams(memdepthParams),
        .memdepthInData(memdepthInData),
        .memdepthTempData1(memdepthTempData1),
        .memdepthTempData2(memdepthTempData2),
        
        .parametersInitFile(memParamsInitFile),
        .inputInitFileSim(memInputInitFileSim)
    )
    
    mem(
        .clk(clk),
        .rst(rst),
        
        .r_data_params(r_data_params),
        .addr_params(addr_params),
        .r_en_params(r_en_params),
        
        .r_data_indata(r_data_indata),
        .addr_indata(addr_indata),
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
    

    wor [addr_width_max-1:0]mem_addr[2:0]; // 0: in1, 1: in2, 2: out
    wor [memwidth-1:0]mem_data[2:0];
    wor mem_en[2:0];
    
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
    reg st_op = 0;
    wand rdy_op;
    
    /// Operation select

    //// st
    localparam VECMATMULT = 3'd0;
    localparam VECVECFUNC = 3'd1;
    localparam VECFUNC = 3'd2;
    localparam ARGMAX = 3'd3;
    
    reg [2:0]op_select = 0;
    wire [5:0]onehot_op_select = st_op << op_select;
    
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
    reg [4:0]vmm_addr_vecin_st_offset = 0;

    ///// vecVecFunc
    reg vvf_disable_addr_vecout = 0;

    ///// vecFunc
    reg [4:0]vf_addr_vecin_st_offset = 0;
    
    
    wor ren_inputs;
    assign mem_en[IN1] = ren_inputs;
    assign mem_en[IN2] = ren_inputs;
    
    /// Vector matrix multiplication
       
    vecMatMult
    #(
        .memwidth(memwidth),
        .elemwidth(elemwidth),
        
        .addr_width_mat(addrwidth_params),
        .addr_width_maxvec(addrwidth_params),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(Qm),
        .Qn(Qn)
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
        
        .rdy(rdy_op),
        
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
        .elemwidth(elemwidth),
        
        .addr_width_max(addr_width_max),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(Qm),
        .Qn(Qn)
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

        .rdy(rdy_op),
        
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
        .elemwidth(elemwidth),
        
        .addr_width_max(addr_width_max),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(Qm),
        .Qn(Qn)
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
        
        .rdy(rdy_op),
        
        .ren_inputs(mem_en[IN1]),
        
        .addr_vecin(mem_addr[IN1]),
        
        .wen_vecout(mem_en[OUT]),
        .w_data_vecout(mem_data[OUT]),
        .addr_vecout(mem_addr[OUT])
    );
    
    /// argmax, find the index of the maximum value in a vector saved in a RAM
    //// NOTE: It is connected directly to tempdata1, as the output data is expected to be there
    
    wire wen_outIndex;
    wire [7:0]argmax_out;
    always @(posedge clk) begin
        if (rst)
            outIndex <= 0;
        else if (wen_outIndex)
            outIndex <= argmax_out;
    end
    
    argmax 
    #(
        .memwidth(memwidth),
        .elemwidth(elemwidth),
        
        .addr_width_max(addrwidth_TempData1),
        
        .memoryReadLatency(memoryReadLatency),
        
        .nIdx(8),
        
        .addr_vecin_st(argmax_vec_st),
        .addr_vecin_end(argmax_vec_end),
        
        .last_vec_size(argmax_last_vec_size)
    )
    
    argmaxblock (
        // Inputs
        .clk(clk),
        .rst(rst),
        
        .st(onehot_op_select[ARGMAX]),
        
        .r_data_vecin(r_data_tempA),
        
        // Outputs
        .rdy(rdy_op),
        
        .ren_inputs(en_tempA),
        .addr_vecin(addr_tempA),
        
        .wen_out(wen_outIndex),
        .index(argmax_out)
    );
    
    
    // ----- Program
    
    reg [7:0]prog_counter = 0;
    reg [7:0]next_prog_counter = 0;
    reg [7:0]num_instr = 25; // 25
    reg en_prog = 0;
    
    localparam IT_MATH_OP = 3'd0; // Perform either vecMatMult, vecVecFunc, vecFunc or argMax
    localparam IT_SET_PC = 3'd1; // Set program counter to a value
    localparam IT_SET_VAR = 3'd2; // Set a variable to a value // TODO: Remove if unused
    reg [2:0]instr_type = 0;
    
    wire prog_end = prog_counter >= num_instr;
    
    /// Variables
    reg [addrwidth_InData - 1:0]addr_indata_offset = feature_size;
    localparam last_addr_indata_offset = feature_size*(num_timesteps-1);
    localparam vecVecFunc_latency = memoryReadLatency+1;
    
    `include "NNprogram.vh"
    
    // FSM
    reg rdy_reg = 1;
    assign rdy = rdy_reg;
   
    localparam WAIT = 4'd0;
    localparam INIT_INSTR = 4'd1;
    localparam LOAD_INSTR = 4'd2;
    localparam SEL_INSTR_TYPE = 4'd3;
    localparam EXECUTE_MATH_OP = 4'd4;
    localparam WAIT_MATH_OP = 4'd5;
    localparam INCR_PC = 4'd6;
    localparam SET_PC = 4'd7;
    localparam END = 4'd8;
    
    reg [3:0]STATE = WAIT;
    
    always @(posedge clk)
    if (rst) begin
        STATE <= WAIT;

        prog_counter <= 0;
        rdy_reg <= 1;
        st_op <= 0;
        en_prog <= 0;
    end
    else
    case (STATE)
        WAIT : begin
            if (st)
                STATE <= INIT_INSTR;
            else
                STATE <= WAIT;

            prog_counter <= 0;
            rdy_reg <= 1;
            st_op <= 0;
            en_prog <= 0;
        end

        INIT_INSTR : begin
            if (prog_end)
                STATE <= END;
            else
                STATE <= LOAD_INSTR;
                
            prog_counter <= prog_counter;
            rdy_reg <= 0;
            st_op <= 0;
            en_prog <= 1;
        end

        LOAD_INSTR : begin
            STATE <= SEL_INSTR_TYPE;
        
            prog_counter <= prog_counter;
            rdy_reg <= 0;
            st_op <= 0;
            en_prog <= 0;
        end
    
        SEL_INSTR_TYPE : begin
            if (instr_type == IT_MATH_OP)
                STATE <= EXECUTE_MATH_OP;
            else if (instr_type == IT_SET_PC)
                STATE <= SET_PC;
            else if (instr_type == IT_SET_VAR)
                STATE <= INCR_PC;
            else // On error
                STATE <= WAIT;
                
            prog_counter <= prog_counter;
            rdy_reg <= 0;
            st_op <= 0;
            en_prog <= 0;
        end

        EXECUTE_MATH_OP : begin
            if (~rdy_op)
                STATE <= WAIT_MATH_OP;
            else
                STATE <= EXECUTE_MATH_OP;
                
            prog_counter <= prog_counter;
            rdy_reg <= 0;
            st_op <= 1;
            en_prog <= 0;
        end

        WAIT_MATH_OP : begin
            if (rdy_op)
                STATE <= INCR_PC;
            else
                STATE <= WAIT_MATH_OP;
                
            prog_counter <= prog_counter;
            rdy_reg <= 0;
            st_op <= 0;
            en_prog <= 0;
        end

        INCR_PC : begin
            STATE <= INIT_INSTR;
        
            prog_counter <= prog_counter + 1;
            rdy_reg <= 0;
            st_op <= 0;
            en_prog <= 0;
        end
        
        SET_PC : begin
            STATE <= INIT_INSTR;
        
            prog_counter <= next_prog_counter;
            rdy_reg <= 0;
            st_op <= 0;
            en_prog <= 0;
        end

        END : begin
            STATE <= WAIT;
            
            prog_counter <= 0;
            rdy_reg <= 0;
            st_op <= 0;
            en_prog <= 0;
        end
        
        default : begin  // Fault Recovery
            STATE <= WAIT;
    
            prog_counter <= 0;
            rdy_reg <= 1;
            st_op <= 0;
            en_prog <= 0;
        end
     endcase
endmodule

