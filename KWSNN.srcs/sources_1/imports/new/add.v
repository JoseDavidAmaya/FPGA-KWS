`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 08:41:27 PM
// Design Name: 
// Module Name: add
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

// Computes a + b
// n = number of bits of a and b
// nn = number of bits of the output

module add #(parameter n=8, parameter nn=n+1)(
    input signed [n-1:0]a,
    input signed [n-1:0]b,
    output signed [nn-1:0]c
    );
    
    assign c = a + b;
endmodule
