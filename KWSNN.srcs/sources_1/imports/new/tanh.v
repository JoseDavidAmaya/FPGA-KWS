`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 08:41:27 PM
// Design Name: 
// Module Name: tanh
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

// Rough PWL approximation of hyperbolic tangent function
// b = 1 if a > 1
// b = -1 if a < -1
// else b = a

// numbers in Qm.n format

module tanh #(parameter m=3, parameter n=4)(
    input signed [m+n:0]a,
    output signed [m+n:0]b
    );
    
    wire signed [m+n:0]m1 = {{(m+1){1'b1}}, {(n){1'b0}}}; // -1
    wire gt1 = a > $signed(1 << n); // Greater than 1
    wire ltm1 = a < $signed(m1); // Less than -1
    
    assign b = gt1 ? $signed(1 << n) : (ltm1 ? m1 : a);
endmodule
