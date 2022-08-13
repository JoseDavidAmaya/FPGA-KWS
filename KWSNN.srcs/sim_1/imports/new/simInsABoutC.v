`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/08/2022 03:48:34 AM
// Design Name: 
// Module Name: simInsABoutC
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

module simInsABoutC();

localparam n = 8;

reg [n-1:0]a = 0;
reg [n-1:0]b = 0;

/*
wire [2*n-1:0]out;
mult #(n) dut(
    .a(a),
    .b(b),
    .c(out)
);
*/


wire [n:0]out;
add #(n) dut(
    .a(a),
    .b(b),
    .c(out)
);


initial begin
    a = -8'sd128;
    b = -8'sd128;
    #(1);
    a = 8'sd127;
    b = -8'sd128;
    #(1);
    a = -8'sd128;
    b = 8'sd127;
    #(1);
    a = 8'sd127;
    b = 8'sd127;
    #(1);
    forever begin
        a = $urandom;
        b = $urandom;
        #(1);
    end
end

endmodule
