`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/20/2022 12:41:30 PM
// Design Name: 
// Module Name: vecVecFunc
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

// Performs elementwise multiplication/addition between two vectors
// 0: Mult, 1: Add

module vecVecFunc
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
        
        input func_select, // 0: Mult, 1: Add
        input disable_addr_vecout, // Set to 1 to output 0 in addr_vecout, useful when we connect a memory both to an input and the output
        
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
    
    reg rdy_reg = 1;
    assign rdy = rdy_reg;
    
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
    
    /// Function
    
    //// Multiplication
    wire [memwidth*2-1:0]tresMult;
    wire [memwidth-1:0]resMult;
    reg [memwidth*2-1:0]tresMult_reg = 0;
    
    always @(posedge clk)
        if (~rdy_reg)
            tresMult_reg <= tresMult;
    
    wire [memwidth-1:0]resMult34;
    mult #(elemwidth) ewMult [m-1:0](
        .a(r_data_vecA),
        .b(r_data_vecB),
        .c(tresMult)
    );

    
    ///// Change Qmn format (We use Q3.4, but the multiplication results in Q7.8, so we change it back)
    Qmn #(2*Qm+1, 2*Qn, Qm, Qn) qmnMult [m-1:0](
        .a(tresMult_reg),
        .b(resMult)
    );
    
    //// Addition
    
    wire [memwidth+m -1:0]tresAdd;
    reg [memwidth+m -1:0]tresAdd_reg = 0;
    
    always @(posedge clk)
        if (~rdy_reg)
            tresAdd_reg <= tresAdd;
    
    wire [memwidth-1:0]resAdd;
    add #(elemwidth) ewAdd [m-1:0](
        .a(r_data_vecA),
        .b(r_data_vecB),
        .c(tresAdd)
    );

    
    sat #(elemwidth+1, elemwidth) satAdd [m-1:0](
        .a(tresAdd_reg),
        .b(resAdd)
    );
    
    localparam MULT = 1'b0;
    localparam ADD = 1'b1;
    
    wire [memwidth-1:0]resFunc = func_select ? resAdd : resMult;
    assign w_data_vecout = datarst ? 0 : resFunc;
    
    /// Vector element iterator
    
    wire [addr_width_max-1:0] addr_base;
    reg [addr_width_max-1:0]addr_base_d[delayCycles-1:0];
    
    always @(posedge clk)
        if (~rdy_reg)
            addr_base_d[0] <= addr_base;
        
    for (i = 1; i < delayCycles; i = i+1) begin
        always @(posedge clk)
            if (~rdy_reg)
                addr_base_d[i] <= addr_base_d[i-1];
    end
    
    wire vecend;
    reg vecend_d[delayCycles-1:0];
    
    always @(posedge clk)
        if (~rdy_reg)
            vecend_d[0] <= vecend;
    
    for (i = 1; i < delayCycles; i = i+1) begin
        always @(posedge clk)
            if (~rdy_reg)
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
    assign addr_vecout = (disable_addr_vecout | datarst) ? 0 : addr_base_d[delayCycles-1] + addr_vecout_st;
    
    
    assign wen_vecout = ~(datarst | datarst_d[delayCycles-1]);
    
    // FSM

   
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
                if (datarst_d[delayCycles-1])
                    STATE <= WAIT;
                else
                    STATE <= END;
    
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