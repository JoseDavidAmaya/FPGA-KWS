`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/20/2022 12:32:30 PM
// Design Name: 
// Module Name: memories
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


module memories
    #(
        parameter memwidth=64,
        parameter memoryReadLatency = 2,
        
        parameter memdepthParams = 1,
        parameter memdepthInData = 1,
        parameter memdepthTempData1 = 1,
        parameter memdepthTempData2 = 1,
        
        parameter parametersInitFile = "params.mem",
        parameter inputInitFileSim = "none",
        
        localparam addrwidth_params = $clog2(memdepthParams),
        localparam addrwidth_InData = $clog2(memdepthInData),
        localparam addrwidth_TempData1 = $clog2(memdepthTempData1),
        localparam addrwidth_TempData2 = $clog2(memdepthTempData2)
    )
    
    (
        input clk,
        input rst,
        
        // Parameters
        output [memwidth-1:0]r_data_params,
        input [addrwidth_params - 1:0]addr_params,
        input r_en_params,
        
        // Input data
        output [memwidth - 1:0]r_data_indata,
        input [addrwidth_InData - 1:0]addr_indata,
        input [memwidth - 1:0]w_data_indata,
        input en_indata,
        input w_en_indata,
        
        // Temporary data / Output data 1
        
        /// Port A
        output [memwidth - 1:0]r_data_tempA,
        input [addrwidth_TempData1 - 1:0]addr_tempA,
        input [memwidth - 1:0]w_data_tempA,
        input en_tempA,
        input w_en_tempA,
        
        /// Port B
        output [memwidth - 1:0]r_data_tempB,
        input [addrwidth_TempData1 - 1:0]addr_tempB,
        input [memwidth - 1:0]w_data_tempB,
        input en_tempB,
        input w_en_tempB,
        
        // Temporary data / Output data 2
        output [memwidth - 1:0]r_data_tempC,
        input [addrwidth_TempData2 - 1:0]addr_tempC,
        input [memwidth - 1:0]w_data_tempC,
        input en_tempC,
        input w_en_tempC
    );
    
    localparam mem_primitive = "distributed"; // Note: when using "distributed", the TDP RAM only has port A available for writing, so when writing the program, port B cannot be used as output
    localparam mem_write_mode = "read_first"; // read_first, no_change, write_first // write_first for better timing
    localparam params_mem_optimization = "true"; // "true" "false"
    
    // ----- Parameters
    
    //`define useIpBlockMem // Choose either block RAM generated from IP catalog, or an XPM (The one from IP catalog has lower utilization)
    
    localparam memParamsBits = memdepthParams*memwidth;
    
    `ifdef useIpBlockMem
    /// Using IP Block Memory generator /// this option uses less block ram than the XPM, however the parameters like memwidth and memdepth have to be updated manually
    params_mem_gen params (
      .clka(clk),    // input wire clka
      .rsta(rst),    // input wire rsta
      .ena(r_en_params),      // input wire ena
      .addra(addr_params),  // input wire [15 : 0] addra
      .douta(r_data_params)  // output wire [127 : 0] douta
    );

    
    `else
    // xpm_memory_sprom: Single Port ROM
    // Xilinx Parameterized Macro, version 2022.1
    
    
    xpm_memory_sprom #(
       .ADDR_WIDTH_A(addrwidth_params),              // DECIMAL
       .AUTO_SLEEP_TIME(0),           // DECIMAL
       .CASCADE_HEIGHT(16),            // DECIMAL
       .ECC_MODE("no_ecc"),           // String
       .MEMORY_INIT_FILE(parametersInitFile),     // String
       .MEMORY_INIT_PARAM("0"),       // String
       .MEMORY_OPTIMIZATION(params_mem_optimization),  // String 
       .MEMORY_PRIMITIVE("block"),     // String
       .MEMORY_SIZE(memParamsBits),            // DECIMAL
       .MESSAGE_CONTROL(0),           // DECIMAL
       .READ_DATA_WIDTH_A(memwidth),        // DECIMAL
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
       .addra(addr_params),                   // ADDR_WIDTH_A-bit input: Address for port A read operations.
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
    `endif
    
    // ----- Input data
    
    localparam memInDataBits = memdepthInData*memwidth;
    
  

    // xpm_memory_spram: Single Port RAM
    // Xilinx Parameterized Macro, version 2022.1
    
    xpm_memory_spram #(
       .ADDR_WIDTH_A(addrwidth_InData),              // DECIMAL
       .AUTO_SLEEP_TIME(0),           // DECIMAL
       .BYTE_WRITE_WIDTH_A(memwidth),       // DECIMAL
       .CASCADE_HEIGHT(0),            // DECIMAL
       .ECC_MODE("no_ecc"),           // String
       .MEMORY_INIT_FILE(inputInitFileSim),     // String
       .MEMORY_INIT_PARAM("0"),       // String
       .MEMORY_OPTIMIZATION("true"),  // String
       .MEMORY_PRIMITIVE(mem_primitive),     // String
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
       .WRITE_MODE_A(mem_write_mode),   // String // TODO: Check the best value here, we don't need to write and read from this memory at the same time, so it might not matter
       .WRITE_PROTECT(1)              // DECIMAL
    )
    indata (
       .dbiterra(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                        // on the data output of port A.
    
       .douta(r_data_indata),                   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
       .sbiterra(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                        // on the data output of port A.
    
       .addra(addr_indata),                   // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
       .clka(clk),                     // 1-bit input: Clock signal for port A.
       .dina(w_data_indata),                     // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
       .ena(en_indata),                       // 1-bit input: Memory enable signal for port A. Must be high on clock
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
       .wea(w_en_indata)                        // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
                                        // for port A input data port dina. 1 bit wide when word-wide writes are
                                        // used. In byte-wide write configurations, each bit controls the
                                        // writing one byte of dina to address addra. For example, to
                                        // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
                                        // is 32, wea would be 4'b0010.
    
    );
    
    // End of xpm_memory_spram_inst instantiation
    
    // ----- Temporary data / Output data 1
    
    localparam memTempData1Bits = memdepthTempData1*memwidth; 
    
    // xpm_memory_tdpram: True Dual Port RAM
    // Xilinx Parameterized Macro, version 2022.1
    
    xpm_memory_tdpram #(
       .ADDR_WIDTH_A(addrwidth_TempData1),               // DECIMAL
       .ADDR_WIDTH_B(addrwidth_TempData1),               // DECIMAL
       .AUTO_SLEEP_TIME(0),            // DECIMAL
       .BYTE_WRITE_WIDTH_A(memwidth),        // DECIMAL
       .BYTE_WRITE_WIDTH_B(memwidth),        // DECIMAL
       .CASCADE_HEIGHT(0),             // DECIMAL
       .CLOCKING_MODE("common_clock"), // String
       .ECC_MODE("no_ecc"),            // String
       .MEMORY_INIT_FILE("none"),      // String // TODO: Change back to none
       .MEMORY_INIT_PARAM("0"),        // String
       .MEMORY_OPTIMIZATION("true"),   // String
       .MEMORY_PRIMITIVE(mem_primitive),      // String
       .MEMORY_SIZE(memTempData1Bits),             // DECIMAL
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
       .WRITE_MODE_A("read_first"),     // String // TODO: Test Allowed values: no_change, read_first, write_first. Default value = no_change
       .WRITE_MODE_B("read_first"),     // String
       .WRITE_PROTECT(1)               // DECIMAL
    )
    tempdata1 (
       .dbiterra(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                        // on the data output of port A.
    
       .dbiterrb(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                        // on the data output of port A.
    
       .douta(r_data_tempA),                   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
       .doutb(r_data_tempB),                   // READ_DATA_WIDTH_B-bit output: Data output for port B read operations.
       .sbiterra(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                        // on the data output of port A.
    
       .sbiterrb(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                        // on the data output of port B.
    
       .addra(addr_tempA),                   // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
       .addrb(addr_tempB),                   // ADDR_WIDTH_B-bit input: Address for port B write and read operations.
       .clka(clk),                     // 1-bit input: Clock signal for port A. Also clocks port B when
                                        // parameter CLOCKING_MODE is "common_clock".
    
       .clkb(),                     // 1-bit input: Clock signal for port B when parameter CLOCKING_MODE is
                                        // "independent_clock". Unused when parameter CLOCKING_MODE is
                                        // "common_clock".
    
       .dina(w_data_tempA),                     // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
       .dinb(w_data_tempB),                     // WRITE_DATA_WIDTH_B-bit input: Data input for port B write operations.
       .ena(en_tempA),                       // 1-bit input: Memory enable signal for port A. Must be high on clock
                                        // cycles when read or write operations are initiated. Pipelined
                                        // internally.
    
       .enb(en_tempB),                       // 1-bit input: Memory enable signal for port B. Must be high on clock
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
       .wea(w_en_tempA),                       // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
                                        // for port A input data port dina. 1 bit wide when word-wide writes are
                                        // used. In byte-wide write configurations, each bit controls the
                                        // writing one byte of dina to address addra. For example, to
                                        // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
                                        // is 32, wea would be 4'b0010.
    
       .web(w_en_tempB)                        // WRITE_DATA_WIDTH_B/BYTE_WRITE_WIDTH_B-bit input: Write enable vector
                                        // for port B input data port dinb. 1 bit wide when word-wide writes are
                                        // used. In byte-wide write configurations, each bit controls the
                                        // writing one byte of dinb to address addrb. For example, to
                                        // synchronously write only bits [15-8] of dinb when WRITE_DATA_WIDTH_B
                                        // is 32, web would be 4'b0010.
    
    );
    
    // End of xpm_memory_tdpram_inst instantiation
    
    
    // ----- Temporary data / Output data 2
    
    localparam memTempData2Bits = memdepthTempData2*memwidth;

    // xpm_memory_spram: Single Port RAM
    // Xilinx Parameterized Macro, version 2022.1
    
    xpm_memory_spram #(
       .ADDR_WIDTH_A(addrwidth_TempData2),              // DECIMAL
       .AUTO_SLEEP_TIME(0),           // DECIMAL
       .BYTE_WRITE_WIDTH_A(memwidth),       // DECIMAL
       .CASCADE_HEIGHT(0),            // DECIMAL
       .ECC_MODE("no_ecc"),           // String
       .MEMORY_INIT_FILE("none"),     // String
       .MEMORY_INIT_PARAM("0"),       // String
       .MEMORY_OPTIMIZATION("true"),  // String
       .MEMORY_PRIMITIVE(mem_primitive),     // String
       .MEMORY_SIZE(memTempData2Bits),            // DECIMAL
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
       .WRITE_MODE_A(mem_write_mode),   // String
       .WRITE_PROTECT(1)              // DECIMAL
    )
    tempdata2 (
       .dbiterra(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                        // on the data output of port A.
    
       .douta(r_data_tempC),                   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
       .sbiterra(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                        // on the data output of port A.
    
       .addra(addr_tempC),                   // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
       .clka(clk),                     // 1-bit input: Clock signal for port A.
       .dina(w_data_tempC),                     // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
       .ena(en_tempC),                       // 1-bit input: Memory enable signal for port A. Must be high on clock
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
       .wea(w_en_tempC)                        // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
                                        // for port A input data port dina. 1 bit wide when word-wide writes are
                                        // used. In byte-wide write configurations, each bit controls the
                                        // writing one byte of dina to address addra. For example, to
                                        // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
                                        // is 32, wea would be 4'b0010.
    
    );
    
    // End of xpm_memory_spram_inst instantiation
    
endmodule
