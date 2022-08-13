`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 08:41:27 PM
// Design Name: 
// Module Name: sigm
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

// Rough PWL approximation of sigmoid function
// A = (a + 1) / 2
// b = 1 if A > 1
// b = 0 if A < 0
// else b = A

// numbers in Qm.n format

module sigm #(parameter m=3, n=4)(
    input signed [m+n:0]a,
    output signed [m+n:0]b
    );
    
    wire signed [m+n:0] A = (a + (1 << n) + 1) >> 1; // (a + 1) / 2
    // we add a 1 (really a 2^-n) so it rounds up
    
    wire gt1 = A > $signed(1 << n); // Greater than 1
    wire lt0 = A < $signed(0); // Less than 0
    
    assign b = gt1 ? (1 << n) : (lt0 ? 0 : A);
endmodule
