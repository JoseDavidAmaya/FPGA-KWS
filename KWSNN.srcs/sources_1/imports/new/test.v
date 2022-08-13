`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/02/2022 05:22:19 PM
// Design Name: 
// Module Name: test
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

// (* dont_touch = "true" *)

module test(
    input clk,
    input rst,
    input st,
    
    input ren_vecout,
    input [$clog2(150) - 1:0]r_addr_vecout,
    output [64-1:0] r_data_vecout,
    
    input wen_indata,
    input [$clog2(250) - 1:0]w_addr_indata,
    input [64-1:0] w_data_indata,
    
    output rdy
    );
    wire [64-1:0] r_data_params;
    wire [64-1:0] r_data_indata;
    wire [64-1:0] w_data_vecout;
    
    localparam memwidth = 64;
    localparam memoryReadLatency = 2;
    
    // Parameters
    
    /// GRU kernel
    localparam memdepth_gk = 6000;
    
    /// GRU recurrent kernel
    localparam memdepth_grk = 60000;
    
    /// GRU bias
    localparam memdepth_gb = 150;
    
    /// FC kernel
    localparam memdepth_fk = 600;
    
    /// FC bias
    localparam memdepth_fb = 2;
    
    localparam memdepthParams = memdepth_gk + memdepth_grk + memdepth_gb + memdepth_fk + memdepth_fb;
    
    wire r_en_params;
    
    
    wire [$clog2(memdepthParams) - 1:0]r_addr_params;
 
    /*
    rom #(memwidth, $clog2(memdepthParams), "C:/Users/jdah1/Desktop/ASR/GRU.hex") params(
        .clk(clk),
        .r_en(r_en_params),
        .r_addr(r_addr_params),
        .r_data(r_data_params)
    );
    */
    
    localparam memParamsBits = memdepthParams*memwidth; // TODO: Change to value calculated in python when exporting
     
    // xpm_memory_sprom: Single Port ROM
    // Xilinx Parameterized Macro, version 2022.1
    
    xpm_memory_sprom #(
       .ADDR_WIDTH_A($clog2(memdepthParams)),              // DECIMAL
       .AUTO_SLEEP_TIME(0),           // DECIMAL
       .CASCADE_HEIGHT(0),            // DECIMAL
       .ECC_MODE("no_ecc"),           // String
       .MEMORY_INIT_FILE("C:/Users/jdah1/Desktop/ASR/GRU.mem"),     // String
       .MEMORY_INIT_PARAM("0"),       // String
       .MEMORY_OPTIMIZATION("true"),  // String
       .MEMORY_PRIMITIVE("block"),     // String
       .MEMORY_SIZE(memParamsBits),            // DECIMAL
       .MESSAGE_CONTROL(0),           // DECIMAL
       .READ_DATA_WIDTH_A(64),        // DECIMAL
       .READ_LATENCY_A(memoryReadLatency),            // DECIMAL
       .READ_RESET_VALUE_A("0"),      // String
       .RST_MODE_A("SYNC"),           // String
       .SIM_ASSERT_CHK(1),            // DECIMAL; 0=disable simulation messages, 1=enable simulation messages
       .USE_MEM_INIT(1),              // DECIMAL
       .USE_MEM_INIT_MMI(0),          // DECIMAL
       .WAKEUP_TIME("disable_sleep")  // String
    )
    params  (
       .dbiterra(),             // 1-bit output: Leave open.
       .douta(r_data_params),                   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
       .sbiterra(),             // 1-bit output: Leave open.
       .addra(r_addr_params),                   // ADDR_WIDTH_A-bit input: Address for port A read operations.
       .clka(clk),                     // 1-bit input: Clock signal for port A.
       .ena(r_en_params),                       // 1-bit input: Memory enable signal for port A. Must be high on clock
                                        // cycles when read operations are initiated. Pipelined internally.
    
       .injectdbiterra(1'b0), // 1-bit input: Do not change from the provided value.
       .injectsbiterra(1'b0), // 1-bit input: Do not change from the provided value.
       .regcea(1'b1),                 // 1-bit input: Do not change from the provided value.
       .rsta(rst),                     // 1-bit input: Reset signal for the final port A output register stage.
                                        // Synchronously resets output port douta to the value specified by
                                        // parameter READ_RESET_VALUE_A.
    
       .sleep(1'b0)                    // 1-bit input: sleep signal to enable the dynamic power saving feature.
    );
    
    // End of xpm_memory_sprom_inst instantiation
    
    
    // Inputs/Outputs/TemporaryValues
    
    // Testing
    
    localparam memdepthInData = 250;
    localparam memInDataBits = memdepthInData*memwidth;
    
    wire [$clog2(memdepthInData) - 1:0]r_addr_indata;
    
    /*
    memory #(memwidth, memdepthInData, "HIGH_PERFORMANCE", "C:/Users/jdah1/Desktop/ASR/qrandomdata.mem") indata(
        .clk(clk),
        .r_en(r_en_params),
        .w_en(wen_indata),
        .r_addr(r_addr_indata),
        .w_addr(w_addr_indata),
        .r_data(r_data_indata),
        .w_data(w_data_indata),
        .out_rst(rst),
        .out_en(r_en_params)
    );
    */
    
    // xpm_memory_spram: Single Port RAM
    // Xilinx Parameterized Macro, version 2022.1
    
    xpm_memory_spram #(
       .ADDR_WIDTH_A($clog2(memdepthInData)),              // DECIMAL
       .AUTO_SLEEP_TIME(0),           // DECIMAL
       .BYTE_WRITE_WIDTH_A(memwidth),       // DECIMAL
       .CASCADE_HEIGHT(0),            // DECIMAL
       .ECC_MODE("no_ecc"),           // String
       .MEMORY_INIT_FILE("C:/Users/jdah1/Desktop/ASR/qrandomdata.mem"),     // String
       .MEMORY_INIT_PARAM("0"),       // String
       .MEMORY_OPTIMIZATION("true"),  // String
       .MEMORY_PRIMITIVE("block"),     // String
       .MEMORY_SIZE(memInDataBits),            // DECIMAL
       .MESSAGE_CONTROL(0),           // DECIMAL
       .READ_DATA_WIDTH_A(memwidth),        // DECIMAL
       .READ_LATENCY_A(memoryReadLatency),            // DECIMAL
       .READ_RESET_VALUE_A("0"),      // String
       .RST_MODE_A("SYNC"),           // String
       .SIM_ASSERT_CHK(1),            // DECIMAL; 0=disable simulation messages, 1=enable simulation messages
       .USE_MEM_INIT(1),              // DECIMAL
       .USE_MEM_INIT_MMI(0),          // DECIMAL
       .WAKEUP_TIME("disable_sleep"), // String
       .WRITE_DATA_WIDTH_A(memwidth),       // DECIMAL
       .WRITE_MODE_A("read_first"),   // String
       .WRITE_PROTECT(1)              // DECIMAL
    )
    inData (
       .dbiterra(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                        // on the data output of port A.
    
       .douta(r_data_indata),                   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
       .sbiterra(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                        // on the data output of port A.
    
       .addra(r_addr_indata),                   // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
       .clka(clk),                     // 1-bit input: Clock signal for port A.
       .dina(w_data_indata),                     // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
       .ena(r_en_params),                       // 1-bit input: Memory enable signal for port A. Must be high on clock
                                        // cycles when read or write operations are initiated. Pipelined
                                        // internally.
    
       .injectdbiterra(1'b0), // 1-bit input: Controls double bit error injection on input data when
                                        // ECC enabled (Error injection capability is not available in
                                        // "decode_only" mode).
    
       .injectsbiterra(1'b0), // 1-bit input: Controls single bit error injection on input data when
                                        // ECC enabled (Error injection capability is not available in
                                        // "decode_only" mode).
    
       .regcea(1'b1),                 // 1-bit input: Clock Enable for the last register stage on the output
                                        // data path.
    
       .rsta(rst),                     // 1-bit input: Reset signal for the final port A output register stage.
                                        // Synchronously resets output port douta to the value specified by
                                        // parameter READ_RESET_VALUE_A.
    
       .sleep(1'b0),                   // 1-bit input: sleep signal to enable the dynamic power saving feature.
       .wea(wen_indata)                        // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
                                        // for port A input data port dina. 1 bit wide when word-wide writes are
                                        // used. In byte-wide write configurations, each bit controls the
                                        // writing one byte of dina to address addra. For example, to
                                        // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
                                        // is 32, wea would be 4'b0010.
    
    );
    
    // End of xpm_memory_spram_inst instantiation
				
				
    
    localparam memdepthVecout = 150;
    localparam memVecoutDataBits = memdepthVecout*memwidth; 
    
    wire [$clog2(memdepthVecout) - 1:0]w_addr_vecout;
    //wire [memwidth - 1:0]w_data_vecout;
    wire w_en_vecout;
    
    
    wire [$clog2(memdepthVecout) - 1:0]w_addr_vecout_b = 0;
    wire w_en_vecout_b = 0;
    wire [memwidth-1:0] r_data_vecout_b;
    wire [memwidth - 1:0]w_data_vecout_b;
    
    /*
    memory #(memwidth, memdepthVecout, "HIGH_PERFORMANCE") outdata(
        .clk(clk),
        .r_en(ren_vecout),
        .w_en(w_en_vecout),
        .r_addr(r_addr_vecout),
        .w_addr(w_addr_vecout),
        .r_data(r_data_vecout),
        .w_data(w_data_vecout),
        .out_rst(rst),
        .out_en(ren_vecout)
    );
    */
    
    
    // xpm_memory_tdpram: True Dual Port RAM
    // Xilinx Parameterized Macro, version 2022.1
    
    xpm_memory_tdpram #(
       .ADDR_WIDTH_A($clog2(memdepthVecout)),               // DECIMAL
       .ADDR_WIDTH_B($clog2(memdepthVecout)),               // DECIMAL
       .AUTO_SLEEP_TIME(0),            // DECIMAL
       .BYTE_WRITE_WIDTH_A(memwidth),        // DECIMAL
       .BYTE_WRITE_WIDTH_B(memwidth),        // DECIMAL
       .CASCADE_HEIGHT(0),             // DECIMAL
       .CLOCKING_MODE("common_clock"), // String
       .ECC_MODE("no_ecc"),            // String
       .MEMORY_INIT_FILE("none"),      // String
       .MEMORY_INIT_PARAM("0"),        // String
       .MEMORY_OPTIMIZATION("true"),   // String
       .MEMORY_PRIMITIVE("block"),      // String
       .MEMORY_SIZE(memVecoutDataBits),             // DECIMAL
       .MESSAGE_CONTROL(0),            // DECIMAL
       .READ_DATA_WIDTH_A(memwidth),         // DECIMAL
       .READ_DATA_WIDTH_B(memwidth),         // DECIMAL
       .READ_LATENCY_A(memoryReadLatency),             // DECIMAL
       .READ_LATENCY_B(memoryReadLatency),             // DECIMAL
       .READ_RESET_VALUE_A("0"),       // String
       .READ_RESET_VALUE_B("0"),       // String
       .RST_MODE_A("SYNC"),            // String
       .RST_MODE_B("SYNC"),            // String
       .SIM_ASSERT_CHK(1),             // DECIMAL; 0=disable simulation messages, 1=enable simulation messages
       .USE_EMBEDDED_CONSTRAINT(0),    // DECIMAL
       .USE_MEM_INIT(1),               // DECIMAL
       .USE_MEM_INIT_MMI(0),           // DECIMAL
       .WAKEUP_TIME("disable_sleep"),  // String
       .WRITE_DATA_WIDTH_A(memwidth),        // DECIMAL
       .WRITE_DATA_WIDTH_B(memwidth),        // DECIMAL
       .WRITE_MODE_A("no_change"),     // String
       .WRITE_MODE_B("no_change"),     // String
       .WRITE_PROTECT(1)               // DECIMAL
    )
    outdata (
       .dbiterra(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                        // on the data output of port A.
    
       .dbiterrb(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                        // on the data output of port A.
    
       .douta(r_data_vecout),                   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
       .doutb(r_data_vecout_b),                   // READ_DATA_WIDTH_B-bit output: Data output for port B read operations.
       .sbiterra(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                        // on the data output of port A.
    
       .sbiterrb(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                        // on the data output of port B.
    
       .addra(w_addr_vecout | r_addr_vecout),                   // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
       .addrb(w_addr_vecout_b),                   // ADDR_WIDTH_B-bit input: Address for port B write and read operations.
       .clka(clk),                     // 1-bit input: Clock signal for port A. Also clocks port B when
                                        // parameter CLOCKING_MODE is "common_clock".
    
       .clkb(),                     // 1-bit input: Clock signal for port B when parameter CLOCKING_MODE is
                                        // "independent_clock". Unused when parameter CLOCKING_MODE is
                                        // "common_clock".
    
       .dina(w_data_vecout),                     // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
       .dinb(w_data_vecout_b),                     // WRITE_DATA_WIDTH_B-bit input: Data input for port B write operations.
       .ena(w_en_vecout | ren_vecout),                       // 1-bit input: Memory enable signal for port A. Must be high on clock
                                        // cycles when read or write operations are initiated. Pipelined
                                        // internally.
    
       .enb(w_en_vecout_b),                       // 1-bit input: Memory enable signal for port B. Must be high on clock
                                        // cycles when read or write operations are initiated. Pipelined
                                        // internally.
    
       .injectdbiterra(1'b0), // 1-bit input: Controls double bit error injection on input data when
                                        // ECC enabled (Error injection capability is not available in
                                        // "decode_only" mode).
    
       .injectdbiterrb(1'b0), // 1-bit input: Controls double bit error injection on input data when
                                        // ECC enabled (Error injection capability is not available in
                                        // "decode_only" mode).
    
       .injectsbiterra(1'b0), // 1-bit input: Controls single bit error injection on input data when
                                        // ECC enabled (Error injection capability is not available in
                                        // "decode_only" mode).
    
       .injectsbiterrb(1'b0), // 1-bit input: Controls single bit error injection on input data when
                                        // ECC enabled (Error injection capability is not available in
                                        // "decode_only" mode).
    
       .regcea(1'b1),                 // 1-bit input: Clock Enable for the last register stage on the output
                                        // data path.
    
       .regceb(1'b1),                 // 1-bit input: Clock Enable for the last register stage on the output
                                        // data path.
    
       .rsta(rst),                     // 1-bit input: Reset signal for the final port A output register stage.
                                        // Synchronously resets output port douta to the value specified by
                                        // parameter READ_RESET_VALUE_A.
    
       .rstb(rst),                     // 1-bit input: Reset signal for the final port B output register stage.
                                        // Synchronously resets output port doutb to the value specified by
                                        // parameter READ_RESET_VALUE_B.
    
       .sleep(1'b0),                   // 1-bit input: sleep signal to enable the dynamic power saving feature.
       .wea(w_en_vecout),                       // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
                                        // for port A input data port dina. 1 bit wide when word-wide writes are
                                        // used. In byte-wide write configurations, each bit controls the
                                        // writing one byte of dina to address addra. For example, to
                                        // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
                                        // is 32, wea would be 4'b0010.
    
       .web(w_en_vecout_b)                        // WRITE_DATA_WIDTH_B/BYTE_WRITE_WIDTH_B-bit input: Write enable vector
                                        // for port B input data port dinb. 1 bit wide when word-wide writes are
                                        // used. In byte-wide write configurations, each bit controls the
                                        // writing one byte of dinb to address addrb. For example, to
                                        // synchronously write only bits [15-8] of dinb when WRITE_DATA_WIDTH_B
                                        // is 32, web would be 4'b0010.
    
    );
    
    // End of xpm_memory_tdpram_inst instantiation
			
    

    // Perform vector matrix multiplication here
    
    vecMatMult
    #(
        .memwidth(memwidth),
        .elemwidth(8),
        .addr_width_mat($clog2(memdepthParams)),
        .addr_width_maxvec($clog2(250)),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(3),
        .Qn(4)
    )
    
    vecmatmultblock(
        .clk(clk),
        .rst(rst),
        .st(st),
         
        .r_data_mat(r_data_params),
        .r_data_vecin(r_data_indata),
        .addr_mat_st(0), // 0 // 6000
        .addr_mat_end(6000), // 6000 // 66000
        .addr_vecin_st(0),
        .addr_vecin_end(5), // 5 // 50
        
        .addr_vecout_st(0),
        .addr_vecout_end(150),
        
        .rdy(rdy),
        .ren_inputs(r_en_params),  
        
        .addr_mat(r_addr_params),
        .addr_vecin(r_addr_indata),
        
        .wen_vecout(w_en_vecout),
        .w_data_vecout(w_data_vecout),
        .addr_vecout(w_addr_vecout)
    );
    
    
    // Peform vector vector multiplication/Addition here
    /*
    vecVecFunc
    #(
        .memwidth(memwidth),
        .elemwidth(8),
        
        .addr_width_max($clog2(memdepthParams)),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(3),
        .Qn(4)
    )
    
    vecvecfunc (
        .clk(clk),
        .rst(rst),
        .st(st),
        
        .func_select(1'b1),
        
        .r_data_vecA(r_data_params),
        .r_data_vecB(r_data_indata),
        
        .addr_vecA_st(0),
        .addr_vecB_st(0),
        
        .addr_vecout_st(0),
        
        .num_positions(10),
        
        .rdy(rdy),
        .ren_inputs(r_en_params),
        
        .addr_vecA(r_addr_params),
        .addr_vecB(r_addr_indata),
        
        .wen_vecout(w_en_vecout),
        .w_data_vecout(w_data_vecout),
        .addr_vecout(w_addr_vecout)
    );
    */
    
    // Perform sigm/tanh/oneMinusA on a vector here
    /*
    vecFunc
    #(
        .memwidth(memwidth),
        .elemwidth(8),
        
        .addr_width_max($clog2(memdepthParams)),
        
        .memoryReadLatency(memoryReadLatency),
        
        .Qm(3),
        .Qn(4)
    )
    
    vecfunc (
        .clk(clk),
        .rst(rst),
        .st(st),
        
        .func_select(2'd0),
        
        .r_data_vecin(r_data_indata),
        
        .addr_vecin_st(0),
        
        .addr_vecout_st(0),
        
        .num_positions(10),
        
        .rdy(rdy),
        .ren_inputs(r_en_params),
        
        .addr_vecin(r_addr_indata),
        
        .wen_vecout(w_en_vecout),
        .w_data_vecout(w_data_vecout),
        .addr_vecout(w_addr_vecout)
    );
    */
    
    /*
    always @(posedge clk) begin
        r_addr_params <= r_addr_params + 1;
        r_addr_indata <= r_addr_indata + 1;
    end
    */
    

endmodule
