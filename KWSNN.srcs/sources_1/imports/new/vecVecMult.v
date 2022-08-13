`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 05:43:29 PM
// Design Name: 
// Module Name: vecVecMult
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

// Computes element-wise product of two vectors

module vecVecMult 
    #(
        parameter memwidth = 64,
        parameter elemwidth = 8,
        
        parameter addr_width_max = 16,
        
        parameter memoryReadLatency = 2,
        
        // Q format of the input data
        parameter Qm = 3,
        parameter Qn = 4
    )

    (
        // Inputs
        input clk,
        input rst,
        
        input st,
        
        input [memwidth-1:0]r_data_vecA,
        input [memwidth-1:0]r_data_vecB,
        
        input [addr_width_max-1:0]addr_vecA_st,
        input [addr_width_max-1:0]addr_vecB_st,
        
        input [addr_width_max-1:0]addr_vecout_st,
        
        input [addr_width_max-1:0]num_positions,
        
        // Outputs
        output rdy,
        
        output reg ren_inputs,
        
        output [addr_width_max-1:0]addr_vecA,
        output [addr_width_max-1:0]addr_vecB,
        
        output wen_vecout,
        output [memwidth-1:0]w_data_vecout,
        output [addr_width_max-1:0]addr_vecout
    );
    
    localparam m = memwidth/elemwidth; // Number of elements read every cycle
    genvar i;
    
    // Data
    localparam delayCycles = memoryReadLatency+1; // (2x default) memoryReadLatency, 1x mult output reg
     
    reg datarst = 1;
    reg datarst_d[delayCycles-1:0];
    
    always @(posedge clk)
        datarst_d[0] <= datarst;
    
    for (i = 1; i < delayCycles; i = i+1) begin
        always @(posedge clk)
            datarst_d[i] <= datarst_d[i-1];
    end
    
    /// Multiplication
    wire [memwidth*2-1:0]resMult;
    reg [memwidth*2-1:0]resMult_reg = 0;
    
    always @(posedge clk)
        resMult_reg <= resMult;
    
    wire [memwidth-1:0]resMult34;
    mult #(elemwidth) ewMult [m-1:0](
        .a(r_data_vecA),
        .b(r_data_vecB),
        .c(resMult)
    );

    
    /// Change Qmn format (We use Q3.4, but the multiplication results in Q7.8, so we change it back)
    Qmn #(2*Qm+1, 2*Qn, Qm, Qn) qmnMult [m-1:0](
        .a(resMult_reg),
        .b(resMult34)
    );
    
    assign w_data_vecout = datarst ? 0 : resMult34;
    
    /// Vector element iterator
    
    wire [addr_width_max-1:0] addr_base;
    reg [addr_width_max-1:0]addr_base_d[delayCycles-1:0];
    
    always @(posedge clk)
        addr_base_d[0] <= addr_base;
        
    for (i = 1; i < delayCycles; i = i+1) begin
        always @(posedge clk)
            addr_base_d[i] <= addr_base_d[i-1];
    end
    
    wire vecend;
    reg vecend_d[delayCycles-1:0];
    
    always @(posedge clk)
        vecend_d[0] <= vecend;
    
    for (i = 1; i < delayCycles; i = i+1) begin
        always @(posedge clk)
            vecend_d[i] <= vecend_d[i-1];
    end
    
    counter #(addr_width_max) vecCounter(
        .clk(clk),
        .rst(datarst),
        .en(1),
        
        .st_val(0),
        .end_val(num_positions),
        .add_val(1),
        
        .val(addr_base),
        .valend(vecend)
    );
   
    assign addr_vecA = datarst ? 0 : addr_base + addr_vecA_st;
    assign addr_vecB = datarst ? 0 : addr_base + addr_vecB_st;
    assign addr_vecout = datarst ? 0 : addr_base_d[delayCycles-1] + addr_vecout_st;
    
    
    assign wen_vecout = ~(datarst | datarst_d[delayCycles-1]);
    
    // FSM
    reg rdy_reg = 1;
    assign rdy = rdy_reg;
   
    localparam WAIT =   2'b00;
    localparam STREAD = 2'b01;
    localparam CALC =   2'b10;
    localparam END =    2'b11;
    
    reg [1:0]STATE = WAIT;
    
    always @(posedge clk)
      if (rst) begin
         STATE <= WAIT;

         rdy_reg <= 1;
         datarst <= 1;
         ren_inputs <= 0;
      end
      else
         case (STATE)
            WAIT : begin
               if (st)
                  STATE <= STREAD;
               else
                  STATE <= WAIT;
                  
                rdy_reg <= 1;
                datarst <= 1;
                ren_inputs <= 0;
            end
            
            STREAD : begin
                STATE <= CALC;
                  
                rdy_reg <= 0;
                datarst <= 1;
                ren_inputs <= 1;
            end
            
            CALC : begin
               if (vecend_d[delayCycles-2])
                  STATE <= END;
               else
                  STATE <= CALC;
    
                rdy_reg <= 0;
                datarst <= 0;
                ren_inputs <= 1;
            end
            
            END : begin
                STATE <= WAIT;
    
                rdy_reg <= 0;
                datarst <= 1;
                ren_inputs <= 0;
            end
            
            default : begin  // Fault Recovery
               STATE <= WAIT;
               
               rdy_reg <= 1;
               datarst <= 1;
               ren_inputs <= 0;
            end
         endcase
    
endmodule
