`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/30/2022 04:58:52 PM
// Design Name: 
// Module Name: argmax
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

// Finds the index of the maximum value in an array of n-bit values that spans multiple addresses in a memory
//      Compared to other operations (vecFunc, vecVecFunc), some parameters are fixed because this operation is only used once

module max #(parameter n=8, parameter nIdx=8)(
    input signed [n-1:0]A,
    input [nIdx-1:0]AIndex,
    input signed [n-1:0]B,
    input [nIdx-1:0]BIndex,
    
    output signed [n-1:0]maxVal,
    output [nIdx-1:0]maxValIndex
    );

    wire maxW = A > B;
    
    assign maxVal = maxW ? A : B;
    assign maxValIndex = maxW ? AIndex : BIndex;
endmodule


module vecArgmax #(parameter n=8, parameter m=8, parameter nIdx=8)(
    input [n*m-1:0]vecA,
    output [n-1:0]maxVal,
    output [nIdx-1:0]index
    );
    
    genvar i, j;
    localparam maxI = m/2;
    localparam maxJ = $clog2(m);
    localparam numWires = m - 1;
    
    wire signed [n-1:0]vecAuf[m-1:0]; // vecA unflattened
    for (i = 0; i < m; i=i+1) begin assign vecAuf[i] = vecA[(i+1)*n-1:i*n]; end // unflatten array
    
    wire signed [n-1:0]tempMax[numWires-1:0];
    wire signed [nIdx-1:0]tempMaxIndex[numWires-1:0];
    
    for(j = 0; j < maxJ; j=j+1) begin : opj
        localparam maxIforJ = 1 << (maxJ - j - 1);
        for(i = 0; i < maxIforJ; i=i+1) begin : opi
            localparam jdisplacein = (1 << (maxJ)) - (1 << (maxJ - j + 1));
            localparam jdisplaceout = (1 << (maxJ)) - (1 << (maxJ - j));
            localparam ii = i*2 + jdisplacein;
            localparam io = i + jdisplaceout;
            if (j == 0) begin
                max #(n, nIdx) maxi(vecAuf[i*2], m-1 - (i*2), vecAuf[i*2+1], m-1 - (i*2+1), tempMax[i], tempMaxIndex[i]);
            end
            else begin
                max #(n, nIdx) maxi(tempMax[ii], tempMaxIndex[ii], tempMax[ii+1], tempMaxIndex[ii+1], tempMax[io], tempMaxIndex[io]);
            end
        end
    end
    
    assign maxVal = tempMax[numWires-1];
    assign index = tempMaxIndex[numWires-1];
endmodule


module vecArgmaxSeg #(parameter n=8, parameter m=8, parameter nIdx=8)(
    input clk,
    input [n*m-1:0]vecA,
    output [n-1:0]maxVal,
    output [nIdx-1:0]index
    );
    
    genvar i, j;
    localparam maxI = m/2;
    localparam maxJ = $clog2(m);
    localparam numWires = m - 1;
    
    wire signed [n-1:0]vecAuf[m-1:0]; // vecA unflattened
    for (i = 0; i < m; i=i+1) begin assign vecAuf[i] = vecA[(i+1)*n-1:i*n]; end // unflatten array
    
    wire signed [n-1:0]tempMax[numWires-1:0];
    reg signed [n-1:0]tempMax_reg[numWires-1:0];
    wire signed [nIdx-1:0]tempMaxIndex[numWires-1:0];
    reg signed [nIdx-1:0]tempMaxIndex_reg[numWires-1:0];
    
    for (i = 0; i < numWires; i=i+1) begin
        always @(posedge clk) begin
            tempMax_reg[i] <= tempMax[i];
            tempMaxIndex_reg[i] <= tempMaxIndex[i];
        end
    end
        
    for(j = 0; j < maxJ; j=j+1) begin : opj
        localparam maxIforJ = 1 << (maxJ - j - 1);
        for(i = 0; i < maxIforJ; i=i+1) begin : opi
            localparam jdisplacein = (1 << (maxJ)) - (1 << (maxJ - j + 1));
            localparam jdisplaceout = (1 << (maxJ)) - (1 << (maxJ - j));
            localparam ii = i*2 + jdisplacein;
            localparam io = i + jdisplaceout;
            if (j == 0) begin
                max #(n, nIdx) maxi(vecAuf[i*2], m-1 - (i*2), vecAuf[i*2+1], m-1 - (i*2+1), tempMax[i], tempMaxIndex[i]);
            end
            else begin
                max #(n, nIdx) maxi(tempMax_reg[ii], tempMaxIndex_reg[ii], tempMax_reg[ii+1], tempMaxIndex_reg[ii+1], tempMax[io], tempMaxIndex[io]);
            end
        end
    end
    
    assign maxVal = tempMax_reg[numWires-1];
    assign index = tempMaxIndex_reg[numWires-1];
endmodule

module argmax 
    #(
        parameter memwidth = 64,
        parameter elemwidth = 8,
        
        parameter addr_width_max = 8,
        
        parameter memoryReadLatency = 2,
        
        parameter nIdx = 8,
        
        parameter addr_vecin_st = 0,
        parameter addr_vecin_end = 2,
        
        parameter last_vec_size = 4
    )
    
    (
        // Inputs
        input clk,
        input rst,
        
        input st,
        
        input [memwidth-1:0]r_data_vecin,
        
        // Outputs
        output rdy,
        
        output reg ren_inputs,
        output [addr_width_max-1:0]addr_vecin,
        
        output reg wen_out,
        output reg [nIdx-1:0]index
    );
    
    localparam m = memwidth/elemwidth; // Number of elements read every cycle
    genvar i;
    
    localparam delayCycles = memoryReadLatency+1+$clog2(m)+1; //(2x default) memoryReadLatency, 1x rdata input reg, (3x default) vecArgMax seg, 1x output reg

    wire vecend;
    reg vecend_d[delayCycles-1:0];
    
    reg datarst = 1;
    reg datarst_d[delayCycles-1:0];
    
    // NOTE: A shift by $clog2(m) multiplies the value by 8, this operation should really be
    // (addr_vecin - addr_vecin_st) * m, however a shift is simpler, but 'm' has to be a power of 2 for this to work
    //wire [nIdx-1:0]indexOffset = (addr_vecin - addr_vecin_st) << 3;
    wire [nIdx-1:0]indexOffset = (addr_vecin - addr_vecin_st) << $clog2(m);
    
    wire last_vec = addr_vecin == (addr_vecin_end-1);
    reg last_vec_d[delayCycles-1:0];
    
    reg [memwidth-1:0]r_data_vecin_reg;
    always @(posedge clk)
        r_data_vecin_reg <= r_data_vecin;
    
    // We set the values of the lastVec (that are all zeroes) to the minimum value so that they are ignored by the vecArgmax
    localparam [elemwidth-1:0]minVal = {1'b1, {(elemwidth-1){1'b0}}};
    localparam [elemwidth-1:0]allZeroes = {(elemwidth){1'b0}};
    localparam non_valid_values = m - last_vec_size;
    localparam [memwidth-1:0]orMask = {{last_vec_size{allZeroes}}, {non_valid_values{minVal}}};

    wire [memwidth-1:0]r_data_vecin_masked = r_data_vecin_reg | orMask;
    
    wire [memwidth-1:0]r_data = last_vec_d[delayCycles-1] ? r_data_vecin_masked : r_data_vecin_reg;
    
    always @(posedge clk) begin
        datarst_d[0] <= datarst;
        vecend_d[0] <= vecend;
        last_vec_d[0] <= last_vec;
    end
    
    for (i = 1; i < delayCycles; i = i+1) begin
        always @(posedge clk) begin
            datarst_d[i] <= datarst_d[i-1];
            vecend_d[i] <= vecend_d[i-1];
            last_vec_d[i] <= last_vec_d[i-1];
        end
    end
    
    wire [addr_width_max-1:0]addr_vecin_;
    assign addr_vecin = datarst ? 0 : addr_vecin_;
    counter #(addr_width_max) vecInAddrCounter(
        .clk(clk),
        .rst(datarst),
        .en(1),
        
        .st_val(addr_vecin_st),
        .end_val(addr_vecin_end),
        .add_val(1),
        
        .val(addr_vecin_),
        .valend(vecend)
    );

    wire [elemwidth-1:0]maxVec;
    //reg [elemwidth-1:0]maxVec_reg;
    wire [nIdx-1:0]idxMaxVec;
    //reg [nIdx-1:0]idxMaxVec_reg;
    
    /*
    always @(posedge clk) begin
        maxVec_reg <= maxVec;
        idxMaxVec_reg <= idxMaxVec;
    end
    */
    
    //vecArgmax #(elemwidth, m, nIdx) vecargmax(
    vecArgmaxSeg #(elemwidth, m, nIdx) vecargmax(
        .clk(clk),
        .vecA(r_data),
        .maxVal(maxVec),
        .index(idxMaxVec)
    );
    
    
    reg [elemwidth-1:0]maxVal_reg;
    wire [elemwidth-1:0]maxVal;
    wire [nIdx-1:0]idxMaxVal;
    always @(posedge clk) begin
        if (datarst_d[delayCycles-1]) begin
            maxVal_reg <= minVal;
            //index <= 0; TODO: Might or might not
        end
        else begin
            maxVal_reg <= maxVal;
            index <= idxMaxVal;
        end
    end
    
    max #(elemwidth, nIdx) tempMax(
        // Which values go in A and B matters, right now it's configured to 
        // not update the index if it finds a value equal to the current max value
        .A(maxVal_reg),
        .AIndex(index),
        .B(maxVec),
        .BIndex(idxMaxVec + indexOffset),
        
        .maxVal(maxVal),
        .maxValIndex(idxMaxVal)
    );
    
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
         wen_out <= 0;
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
                wen_out <= 0;
            end
            
            STREAD : begin
                STATE <= CALC;
                  
                rdy_reg <= 0;
                datarst <= 1;
                ren_inputs <= 1;
                wen_out <= 0;
            end
            
            CALC : begin
               if (vecend_d[delayCycles-2])
                  STATE <= END;
               else
                  STATE <= CALC;
    
                rdy_reg <= 0;
                datarst <= 0;
                ren_inputs <= 1;
                wen_out <= 0;
            end
            
            END : begin
                /*
                if (datarst_d[delayCycles-1])
                    STATE <= WAIT;
                else
                    STATE <= END;
                */
                STATE <= WAIT;
    
                rdy_reg <= 0;
                datarst <= 1;
                ren_inputs <= 0;
                wen_out <= 1;
            end
            
            default : begin  // Fault Recovery
               STATE <= WAIT;
               
               rdy_reg <= 1;
               datarst <= 1;
               ren_inputs <= 0;
               wen_out <= 0;
            end
         endcase
    
endmodule
