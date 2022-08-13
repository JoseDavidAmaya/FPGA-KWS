`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/20/2022 05:51:11 AM
// Design Name: 
// Module Name: vecFunc
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

// Applies a function to each element of a vector
// 0: Sigm, 1: Tanh, 2: 1 - A

module vecFunc
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
        
        input [1:0]func_select, // 0: Sigm, 1: Tanh, 2: 1 - A
        
        input [memwidth-1:0]r_data_vecin,
        
        input [addr_width_max-1:0]addr_vecin_st,
        input [3:0]addr_vecin_st_offset,
        
        input [addr_width_max-1:0]addr_vecout_st,
        
        input [addr_width_max-1:0]num_positions,
        
        // Outputs
        output rdy,
        
        output reg ren_inputs,
        
        output [addr_width_max-1:0]addr_vecin,
        
        output wen_vecout,
        output [memwidth-1:0]w_data_vecout,
        output [addr_width_max-1:0]addr_vecout
    );
    
    localparam m = memwidth/elemwidth; // Number of elements read every cycle
    genvar i;
    
    reg rdy_reg = 1;
    assign rdy = rdy_reg;
    
    // Data
    localparam outDelayCycles = 3;
    localparam delayCycles = memoryReadLatency+1+outDelayCycles; // (2x default) memoryReadLatency, 1x output reg, (addr_delayCycles)x addr_vecA|B delay for timing
    
    reg datarst = 1;
    reg datarst_d[delayCycles-1:0];
    
    always @(posedge clk)
        datarst_d[0] <= datarst;
    
    for (i = 1; i < delayCycles; i = i+1) begin
        always @(posedge clk)
            datarst_d[i] <= datarst_d[i-1];
    end
    
    /// Function
    // Func select
    localparam SIGM = 2'd0;
    localparam TANH = 2'd1;
    localparam ONEMA = 2'd2;
    
    wire [memwidth-1:0]resSigm;
    wire [memwidth-1:0]resTanh;
    wire [memwidth-1:0]resOneMA;
    reg [memwidth-1:0]resFunc_reg = 0;
        
    sigm #(Qm, Qn) sigmFunc [m-1:0](
        .a(r_data_vecin),
        .b(resSigm)
    );

    tanh #(Qm, Qn) tanhFunc [m-1:0](
        .a(r_data_vecin),
        .b(resTanh)
    );

    oneMinusA #(Qm, Qn) oneMinusAFunc [m-1:0](
        .a(r_data_vecin),
        .b(resOneMA)
    );

    always @(posedge clk)
        if (~rdy_reg)
            case (func_select)
                SIGM: resFunc_reg <= resSigm;
                TANH: resFunc_reg <= resTanh;
                ONEMA: resFunc_reg <= resOneMA;
                
                default: resFunc_reg <= 0;
            endcase
    
    
    
    /// Vector element iterator
    
    wire [addr_width_max-1:0]addr_base;
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
    reg vecend_d[delayCycles+outDelayCycles-1:0];
    
    always @(posedge clk)
        if (~rdy_reg)
            vecend_d[0] <= vecend;
    
    for (i = 1; i < delayCycles+outDelayCycles; i = i+1) begin
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
   
    wire [addr_width_max:0]addr_vecin_offset = addr_base + addr_vecin_st + addr_vecin_st_offset;
    wire [addr_width_max:0]addr_vecin_offset_wrap = addr_vecin_offset - num_positions;
    wire wrap_around = addr_vecin_offset >= (addr_vecin_st + num_positions);
    
    wire [addr_width_max-1:0]addr_vecin_ = (wrap_around ? addr_vecin_offset_wrap[addr_width_max-1:0] : addr_vecin_offset[addr_width_max-1:0]);
    wire [addr_width_max-1:0]addr_vecout_ = addr_base_d[delayCycles-1] + addr_vecout_st;
    

    reg [addr_width_max-1:0]addr_vecin_d[outDelayCycles-1:0];
    reg [addr_width_max-1:0]addr_vecout_d[outDelayCycles-1:0];
    reg [memwidth-1:0]resFunc_reg_d[outDelayCycles-1:0];
    
    for (i = 0; i < outDelayCycles; i=i+1) begin
        always @(posedge clk) begin
            if (i == 0) begin
                addr_vecin_d[i] <= addr_vecin_;
                addr_vecout_d[i] <= addr_vecout_;
                resFunc_reg_d[i] <= resFunc_reg;
            end
            else begin
                addr_vecin_d[i] <= addr_vecin_d[i-1];
                addr_vecout_d[i] <= addr_vecout_d[i-1];
                resFunc_reg_d[i] <= resFunc_reg_d[i-1];
            end
        end
    end
    
    assign addr_vecin = datarst ? 0 : addr_vecin_d[outDelayCycles-1];
    assign addr_vecout = datarst ? 0 : addr_vecout_d[outDelayCycles-1];
    assign w_data_vecout = datarst ? 0 : resFunc_reg_d[outDelayCycles-1];
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
               if (vecend_d[delayCycles+outDelayCycles-2])
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
