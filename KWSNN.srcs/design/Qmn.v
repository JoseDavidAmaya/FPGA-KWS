`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/08/2022 12:40:04 AM
// Design Name: 
// Module Name: Qmn
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

// Converts a Qm1.n1 number to Qm2.n2
// Restriction: n1 > n2

//(* dont_touch = "true" *)
module Qmn #(parameter mIn=7, parameter nIn=8, parameter mOut=3, parameter nOut=4)( // mIn = m1 ...
    input signed [mIn+nIn:0]a,
    output signed [mOut+nOut:0]b
    );

    wire [mIn+nIn:0]tempB = (a + (1 << (nIn - nOut - 1))) >>> (nIn - nOut);
    sat #((mIn+nIn)+1, (mOut+nOut)+1) sat1(
        .a(tempB),
        .b(b)
    );
endmodule
