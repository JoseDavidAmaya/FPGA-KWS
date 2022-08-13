`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 11:55:38 PM
// Design Name: 
// Module Name: oneMinusA
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

// b = 1 - a
// numbers in Qm.n format 

module oneMinusA #(parameter m=3, parameter n=4)(
    input signed [m+n:0]a,
    output signed [m+n:0]b
    );
    
    wire signed [m+n+1:0]tempB = $signed(1 << n) - a;
    sat #((m+n+1)+1, (m+n)+1) sat1(
        .a(tempB),
        .b(b)
    );
    // Saturate
   
    //wire [m+n:0]maxVal = {1'b0, {(m+n){1'b1}}};
    //assign b = ($signed(tempB) > $signed({maxVal})) ? maxVal : tempB[m+n:0]; 

endmodule