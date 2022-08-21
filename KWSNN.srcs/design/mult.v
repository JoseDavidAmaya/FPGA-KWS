`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 05:06:17 PM
// Design Name: 
// Module Name: mult
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

// Computes a * b
// n = number of bits of a and b

module mult #(parameter n=8)(
    input signed [n-1:0]a,
    input signed [n-1:0]b,
    output signed [2*n-1:0]c
    );
    
    assign c = a * b;
    
endmodule
