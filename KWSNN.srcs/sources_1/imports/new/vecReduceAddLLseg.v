`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/14/2022 06:14:00 PM
// Design Name: 
// Module Name: vecReduceAddLLseg
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


// Segmented, with a register after every 'add' or 'sat' block
//  For m = 8, the latency is 6 clock cycles

// TODO: Add clock enable to registers to save power
// ^ Needs a CE input
module vecReduceAddLLseg #(parameter n=8, parameter m=8, parameter nn=n+1)(
    input clk,
    input [n*m-1:0]vecA,
    output signed [nn-1:0]vecB
    );
    
    genvar i, j;
    localparam maxI = m/2;
    localparam maxJ = $clog2(m);
    localparam numWires = m - 1;
       
    wire signed [n-1:0]vecAuf[m-1:0]; // vecA unflattened
    for (i = 0; i < m; i=i+1) begin assign vecAuf[i] = vecA[(i+1)*n-1:i*n]; end // unflatten array
    
    wire signed [nn:0]tempA[numWires-1:0];
    reg signed [nn:0]tempA_reg[numWires-1:0];
    wire signed [nn-1:0]tempAs[numWires-1:0]; // After saturation
    reg signed [nn-1:0]tempAs_reg[numWires-1:0]; // After saturation
    
    for (i = 0; i < numWires; i=i+1) begin
        always @(posedge clk) begin
            tempA_reg[i] <= tempA[i];
            tempAs_reg[i] <= tempAs[i];
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
                (* keep_hierarchy = "true" *) add #(n, nn+1) addi(vecAuf[i*2], vecAuf[i*2+1], tempA[i]);
                sat #(nn+1, nn) sati(tempA_reg[i], tempAs[i]);     
            end
            else begin
                (* keep_hierarchy = "true" *) add #(nn, nn+1) addi(tempAs_reg[ii], tempAs_reg[ii+1], tempA[io]);
                sat #(nn+1, nn) sati(tempA_reg[io], tempAs[io]);
            end
        end
    end
        
    assign vecB = tempAs_reg[numWires-1];
    
endmodule