`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/12/2022 04:40:37 PM
// Design Name: 
// Module Name: vecReduceAddLL
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

// 'low latency' version of vecReduceAdd
// 'm' has to be a power of two

`define signExtend(val, nIn, nOut) {{(nOut-nIn){val[nIn-1]}}, val}

module vecReduceAddLL #(parameter n=8, parameter m=8, parameter nn=n+1)(
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
    wire signed [nn-1:0]tempAs[numWires-1:0]; // After saturation
    
    for(j = 0; j < maxJ; j=j+1) begin : opj
        localparam maxIforJ = 1 << (maxJ - j - 1);
        for(i = 0; i < maxIforJ; i=i+1) begin : opi
            localparam jdisplacein = (1 << (maxJ)) - (1 << (maxJ - j + 1));
            localparam jdisplaceout = (1 << (maxJ)) - (1 << (maxJ - j));
            localparam ii = i*2 + jdisplacein;
            localparam io = i + jdisplaceout;
            if (j == 0) begin
                add #(n, nn+1) addi(vecAuf[i*2], vecAuf[i*2+1], tempA[i]);
                sat #(nn+1, nn) sati(tempA[i], tempAs[i]);     
            end
            else begin
                add #(nn, nn+1) addi(tempAs[ii], tempAs[ii+1], tempA[io]);
                sat #(nn+1, nn) sati(tempA[io], tempAs[io]);
            end
        end
    end
        
    assign vecB = tempAs[numWires-1];
    
endmodule
