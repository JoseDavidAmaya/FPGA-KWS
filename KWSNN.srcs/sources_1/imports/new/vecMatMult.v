`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/10/2022 12:19:18 AM
// Design Name: 
// Module Name: vecMatMult
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

// Performs vector-matrix multiplication

module vecMatMult
    #(
        parameter memwidth = 64,
        parameter elemwidth = 8,
        
        parameter addr_width_mat = 16,
        parameter addr_width_maxvec = 16,
        
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
        
        
        input [memwidth-1:0]r_data_mat,
        input [memwidth-1:0]r_data_vecin,
        
        input [addr_width_mat-1:0]addr_mat_st,
        input [addr_width_mat-1:0]addr_mat_end,
        input [addr_width_maxvec-1:0]addr_vecin_st,
        input [addr_width_maxvec-1:0]addr_vecin_end,
        input [3:0]addr_vecin_st_offset,
        
        input [addr_width_maxvec-1:0]addr_vecout_st,
        input [addr_width_maxvec-1:0]addr_vecout_end,
        
        // Outputs
        output rdy,
        
        output reg ren_inputs,
        
        output [addr_width_mat-1:0]addr_mat,
        output [addr_width_maxvec-1:0]addr_vecin,
        
        output wen_vecout,
        output [memwidth-1:0]w_data_vecout,
        output [addr_width_maxvec-1:0]addr_vecout
    );
    
    localparam m = memwidth/elemwidth; // Number of elements read every cycle
    genvar i;
    
    reg rdy_reg = 1;
    assign rdy = rdy_reg;
    
    // Data
    reg datarst = 1;
    
    /// Dot product
    wire [memwidth*2-1:0] dotMult;
    mult #(elemwidth) vecVecMultTemp [m-1:0](
        .a(r_data_vecin),
        .b(r_data_mat),
        .c(dotMult)
    );
    
    localparam addExtraBits = 1; // Number of extra bits to use in the vecReduceAdd operation, 
    wire [elemwidth*2+addExtraBits -1:0]dotAdd;
    reg [elemwidth*2+addExtraBits -1:0]dotAdd_reg = 0;
    
    reg [memwidth*2-1:0]dotMult_reg = 0;
   
    
    //vecReduceAdd #(elemwidth*2, m, elemwidth*2+addExtraBits) vecReduceAddTemp(
    //vecReduceAddLL #(elemwidth*2, m, elemwidth*2+addExtraBits) vecReduceAddLLTemp(
    vecReduceAddLLseg #(elemwidth*2, m, elemwidth*2+addExtraBits) vecReduceAddLLsegTemp(
        .clk(clk),
        .vecA(dotMult_reg),
        .vecB(dotAdd)
    );
    
    /// Temp adder
    localparam delayCycles = memoryReadLatency+2+($clog2(m)*2); // (2x default) memoryReadLatency, 1x reduceAdd input reg, 1x reduceAdd output reg, (6x default) vecReduceAddLLseg segmentation
    wire change_column; // Section: //// Input vector elements iterator
    reg change_column_d[delayCycles:0]; // Section: //// Input vector elements iterator
    
    always @(posedge clk)
        if (~rdy_reg)
            change_column_d[0] <= change_column;
        
    for (i = 1; i < delayCycles+1; i=i+1) begin
        always @(posedge clk) 
            if (~rdy_reg)
                change_column_d[i] <= change_column_d[i-1];
    end
    
    
    reg [elemwidth*2+addExtraBits -1:0]tempRes;
    wire [elemwidth*2+addExtraBits+1 -1:0]trA; //[m-1:0];
    wire [elemwidth*2+addExtraBits -1:0]trS; //[m-1:0];
    
    reg [elemwidth -1:0]tempResSR[m-1:0]; // Shift register
    
    wire [elemwidth*m -1:0]tempResFlat;
    for (i = 0; i < m; i=i+1) begin assign tempResFlat[i*(elemwidth)+:elemwidth] = tempResSR[i]; end // flatten array
    
    //// Change Qmn format (We use Q3.4, but the multiplication and addition results in Q8.8 (with default settings), so we change it back)

    wire [elemwidth -1:0]tempRes34;

    Qmn #(Qm*2+1+addExtraBits, Qn*2, Qm, Qn) qmntr( 
        .a(tempRes),
        .b(tempRes34)
    );
    
    for (i = 0; i < m; i=i+1) begin 
        always @(posedge clk)
            if (~rdy_reg) begin
                if (change_column_d[delayCycles]) 
                    if (i == 0)
                        tempResSR[i] <= tempRes34;
                    else
                        tempResSR[i] <= tempResSR[i-1];
            end
    end
    
    wire write_out; // Section: //// Temporary output vector iterator
    reg write_out_d = 0; // Section: //// Temporary output vector iterator
    wire write_out_s = write_out_d & ~write_out; // makes write_out last a single clock cycle // Section: //// Temporary output vector iterator
    reg write_out_sd;
    
    reg datarst_d[delayCycles-1:0];
    
    always @(posedge clk) begin // Section: //// Temporary output vector iterator
        if (~rdy_reg) begin
            write_out_d <= write_out;
            write_out_sd <= write_out_s;
        end
        
        datarst_d[0] <= datarst;
    end
    
    for (i = 1; i < delayCycles; i=i+1) begin
        always @(posedge clk) begin
            datarst_d[i] <= datarst_d[i-1];
        end
    end
 
    always @(posedge clk) begin // Dot product
        if (~rdy_reg) begin
            dotAdd_reg <= dotAdd;
            dotMult_reg <= dotMult;
        end
    end
       
    add #(elemwidth*2+addExtraBits) addi(dotAdd_reg, tempRes, trA);
    sat #(elemwidth*2+addExtraBits+1, elemwidth*2+addExtraBits) sati(trA, trS);
    
    always @(posedge clk) begin
        if (datarst_d[delayCycles-1])
            tempRes <= 0;
        else if (change_column_d[delayCycles])
            tempRes <= dotAdd_reg;
        else 
            tempRes <= trS;
    end

    /// Iterators
   
    //// Input vector elements iterator
    wire [addr_width_maxvec-1:0]addr_vecin_;
    
    wire [addr_width_maxvec:0]addr_vecin_offset = addr_vecin_ + addr_vecin_st_offset;
    wire [addr_width_maxvec:0]addr_vecin_offset_wrap = addr_vecin_offset + addr_vecin_st - addr_vecin_end ;
    wire wrap_around = addr_vecin_offset >= addr_vecin_end;
    
    assign addr_vecin = datarst ? 0 : (wrap_around ? addr_vecin_offset_wrap[addr_width_maxvec-1:0] : addr_vecin_offset[addr_width_maxvec-1:0]);
    counter #(addr_width_maxvec) vecInCounter(
        .clk(clk),
        .rst(datarst),
        .en(1),
        
        .st_val(addr_vecin_st),
        .end_val(addr_vecin_end),
        .add_val(1),
        
        .val(addr_vecin_),
        .valend(change_column)
    );
    
    //// Matrix elements iterator
    wire [addr_width_mat-1:0]addr_mat_;
    assign addr_mat = datarst ? 0 : addr_mat_;
    counter #(addr_width_mat) matCounter(
        .clk(clk),
        .rst(datarst),
        .en(1),
        
        .st_val(addr_mat_st),
        .end_val(addr_mat_end),
        .add_val(1),
        
        .val(addr_mat_),
        .valend()
    );
    
    //// Temporary output vector iterator
    //// (We calculate m values with width elemwidth so that we can write to the output vector memmory which has width memwidth = m * elemwidth)

    counter #($clog2(m)+1) tempResCounter(
        .clk(clk),
        .rst(datarst_d[delayCycles-1]),
        .en(change_column_d[delayCycles-1]),
        
        .st_val(0),
        .end_val(m),
        .add_val(1),
        
        .val(),
        .valend(write_out)
    );
    
    //// Output vector elements address iterator
    wire [addr_width_maxvec-1:0]addr_vecout_;
    reg [addr_width_maxvec-1:0]addr_vecout_d[delayCycles-1:0];
    
    always @(posedge clk)
        if (~rdy_reg)
            addr_vecout_d[0] <= addr_vecout_;
        
    for (i = 1; i < delayCycles; i=i+1) begin
        always @(posedge clk)
            if (~rdy_reg)
                addr_vecout_d[i] <= addr_vecout_d[i-1];
    end
    
    assign addr_vecout = datarst ? 0 : addr_vecout_d[delayCycles-1];
    wire vend;
    reg vend_d = 0;
    wire vend_s = vend_d & ~vend;
    always @(posedge clk)
        if (~rdy_reg)
            vend_d <= vend;
    
    
    counter #(addr_width_maxvec) vecOutCounter(
        .clk(clk),
        .rst(datarst),
        .en(write_out_s),
        
        .st_val(addr_vecout_st),
        .end_val(addr_vecout_end),
        .add_val(1),
        
        .val(addr_vecout_),
        .valend(vend)
    );
    
    assign w_data_vecout = datarst ? 0 : tempResFlat;
    assign wen_vecout = datarst ? 0 : write_out_sd;
    
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
                datarst <= 0;
                ren_inputs <= 1;
            end
            
            CALC : begin
               if (vend_d & write_out_sd) // vend_d & write_out_sd // vend_s
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

