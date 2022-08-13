`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/03/2022 02:52:26 PM
// Design Name: 
// Module Name: NN_Controller
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


module NN_Controller(
        input clk,
        input rst,
        
        // SPI Interface
        input SCK,
        input CS,
        input MOSI,
        output MISO,
        
        // Seven Segment Display
        output [6:0]sevenSegment,
        output [7:0]segment,
        
        output rdy
    );
    
    // FSM regs
    reg rdy_reg = 1; assign rdy = rdy_reg;
    reg datarst = 1;
    reg inData_writing = 0;
    reg st_nn = 0;
    reg disp_idx = 0;
    reg ren_spi = 1;
    // --
    
    // SPI Interface
    wire [7:0]data_miso;
    wire [7:0]data_mosi;
    wire data_mosi_ready;
    
    spi_peripheral spiblock(
        .clk(clk),
        .rst(rst),
        
        .ren(ren_spi),
        .data_mosi_ready(data_mosi_ready),
        
        .data_miso(data_miso),
        .data_mosi(data_mosi),
        
        .SCK(SCK),
        .MOSI(MOSI),
        .MISO(MISO),
        .CS(CS)
    );
    
    
    /// To write to the inData memmory we use a shift register to load 8 values before writing to it
    `include "NNparams.vh"
    
    /// Shift register, loads 8 8-bit values before writing them to the memory
    reg [memwidth-1:0]memSR;
    
    always @(posedge clk)
        if(data_mosi_ready & inData_writing)
            memSR <= {memSR[memwidth-1 -elemwidth:0], data_mosi}; // Shifts 8 bits
    
    
    /// Counter to check if the SR has been completely updated with new values
    wand write_indata;
    reg write_indata_d;
    assign write_indata = data_mosi_ready;
    always @(posedge clk)
        write_indata_d <= write_indata;
    
    counter #($clog2(m)+1) shiftsCounter(
        // Inputs
        .clk(clk),
        .rst(datarst),
        .en(data_mosi_ready & inData_writing),
        
        .st_val(0),
        .end_val(m),
        .add_val(1),
        
        // Outputs
        .val(),
        .valend(write_indata)
    );
    
    /// Counter to change the address we are writing to, also signals when we have finished writing
    
    //// inData memory inputs
    wire [addrwidth_InData-1:0]addr_inData;
    wire [memwidth - 1:0]data_indata;
    wire en_indata;
    wire w_en_indata;
    //// --
    
    wire write_end;
    
    reg write_end_d = 0;
    wire write_end_s = write_end_d & ~write_end;
    always @(posedge clk)
        write_end_d <= write_end;
    
    counter #(addrwidth_InData) addrCounter(
        // Inputs
        .clk(clk),
        .rst(datarst),
        .en(write_indata_d),
        
        .st_val(0),
        .end_val(memdepthInData),
        .add_val(1),
        
        // Outputs
        .val(addr_inData),
        .valend(write_end)
    );
    
    assign data_indata = memSR;
    assign en_indata = write_indata_d;
    assign w_en_indata = write_indata_d;
    
    // NN Processor
    
    reg [7:0]index = 11;
    wire [7:0]index_;
    always @(posedge clk)
        if (disp_idx)
            index <= index_;
    
    assign data_miso = index; // We read the index (Output of the NN) through the SPI
    
    wire rdy_nn;
    
    NN nnblock(
        .clk(clk),
        .rst(datarst), // TODO: Maybe datarst?
        .st(st_nn),
        
        .access_addr_indata(addr_inData),
        .access_w_data_indata(data_indata),
        .access_en_indata(en_indata),
        .access_w_en_indata(w_en_indata),
        
        .outIndex(index_),
        .rdy(rdy_nn)
    );
    
    // Seven segment display
    sevenSegLabelShow display(
        .clk(clk),
        .rst(rst),
        .idx(index[3:0]),
        
        .sevenSegment(sevenSegment),
        .segment(segment)
    );
    
    
    // FSM
    
    /// To start writing to the memory, we first send 'STW' from the PC to the FPGA, and after that we send the data
    reg [2:0]stwIdx = 0;
    wire [2:0]stwEnd = 3;
    reg [7:0]stw[2:0];
    
    initial begin
        stw[0] = 8'h53; // 'S' in ASCII 
        stw[1] = 8'h54; // 'T' in ASCII
        stw[2] = 8'h57; // 'W' in ASCII
    end
    
    always @(posedge clk)
        if (stwIdx == stwEnd)
            stwIdx <= 0;
        else if (data_mosi_ready)
            stwIdx <= ( data_mosi == stw[stwIdx] ) ? ( stwIdx + 1 ) : ( data_mosi == stw[0] ? 1 : 0 );
    /// --
       
    localparam WAIT =        3'd0;
    localparam START_WRITE = 3'd1;
    localparam WAIT_WRITE =  3'd2;
    localparam START_NN =    3'd3;
    localparam WAIT_NN =     3'd4;
    localparam DISP =        3'd5;
    
    reg [2:0]STATE = WAIT;
    
    always @(posedge clk)
      if (rst) begin
         STATE <= WAIT;

         rdy_reg <= 1;
         datarst <= 1;
         inData_writing <= 0;
         st_nn <= 0;
         disp_idx <= 0;
         ren_spi <= 1;
      end
      else
         case (STATE)
            WAIT : begin
               if (stwIdx == stwEnd)
                  STATE <= START_WRITE;
               else
                  STATE <= WAIT;
                  
                rdy_reg <= 1;
                datarst <= 1;
                inData_writing <= 0;
                st_nn <= 0;
                disp_idx <= 0;
                ren_spi <= 1;
            end
            
            START_WRITE : begin
                STATE <= WAIT_WRITE;
                  
                rdy_reg <= 0;
                datarst <= 0;
                inData_writing <= 1;
                st_nn <= 0;
                disp_idx <= 0;
                ren_spi <= 0;
            end
            
            WAIT_WRITE : begin
               if (write_end_s)
                  STATE <= START_NN;
               else
                  STATE <= WAIT_WRITE;
    
                rdy_reg <= 0;
                datarst <= 0;
                inData_writing <= 1;
                st_nn <= 0;
                disp_idx <= 0;
                ren_spi <= 0;
            end
            
            START_NN: begin
               if (~rdy_nn)
                  STATE <= WAIT_NN;
               else
                  STATE <= START_NN;
               
                rdy_reg <= 0;
                datarst <= 0;
                inData_writing <= 0;
                st_nn <= 1;
                disp_idx <= 0;
                ren_spi <= 0;
            end
            
            WAIT_NN: begin
               if (rdy_nn)
                  STATE <= DISP;
               else
                  STATE <= WAIT_NN;
               
                rdy_reg <= 0;
                datarst <= 0;
                inData_writing <= 0;
                st_nn <= 0;
                disp_idx <= 0;
                ren_spi <= 0;
            end
            
            DISP: begin
                STATE <= WAIT;
                
                rdy_reg <= 0;
                datarst <= 0;
                inData_writing <= 0;
                st_nn <= 0;
                disp_idx <= 1;
                ren_spi <= 0;
            end
            
            default : begin  // Fault Recovery
               STATE <= WAIT;
               
               rdy_reg <= 1;
               datarst <= 1;
               inData_writing <= 0;
               st_nn <= 0;
               disp_idx <= 0;
               ren_spi <= 1;
            end
         endcase
    
endmodule
