`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/08/2022 05:53:21 AM
// Design Name: 
// Module Name: vecReduceAdd
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

// Computes the sum of all the elements in a vector
// n = number of bits of each element of veca
// m = number of elements of veca
// nn = number of bits of the temporary result, affects accuracy

// Note: to have the correct result, in the output we need 'n+m-1' bits, however 'm' might be number bigger than 100 (if operations are done in parallel)
// and we don't have that much space, so we define 'nn' as the size of the temporary value to which we will add numbers to
// and apply saturation every addition. Based on the data we are working on, there's little possibility of an overflow, however
// nn can be increased until the accuracy required is reached at the cost of a higher resource utilization

`define signExtend(val, nIn, nOut) {{(nOut-nIn){val[nIn-1]}}, val}

module vecReduceAdd #(parameter n=8, parameter m=8, parameter nn=n+1)(
    input [n*m-1:0]vecA,
    output signed [nn-1:0]vecB
    );
    
    genvar i;
    
    wire signed [n-1:0]vecAuf[m-1:0]; // vecA unflattened
    for (i = 0; i < m; i=i+1) begin assign vecAuf[i] = vecA[(i+1)*n-1:i*n]; end // unflatten array
    
    wire signed [nn:0]tempA[m-2:0];
    wire signed [nn-1:0]tempAs[m-2:0]; // After saturation
    
    // tempA_i = tempAs_i-1 + a_i+1
    // tempAs_i = sat(tempA_i)
    
    add #(n, nn+1) add01(vecAuf[0], vecAuf[1], tempA[0]);
    sat #(nn+1, nn)sat0 (tempA[0], tempAs[0]);
    for(i = 1; i < m-1; i=i+1) begin : opi
        add #(nn, nn+1) addi(tempAs[i-1], `signExtend(vecAuf[i+1], n, nn), tempA[i]); // we sign extend the inputs (they have 'n' bits and have to be 'nn' bits)
        sat #(nn+1, nn) sati(tempA[i], tempAs[i]);
    end
    
    assign vecB = tempAs[m-2];
    
endmodule
