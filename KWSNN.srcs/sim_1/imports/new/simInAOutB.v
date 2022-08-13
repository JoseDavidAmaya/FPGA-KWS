`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 08:54:58 PM
// Design Name: 
// Module Name: simInAOutB
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


module simInAOutB();


localparam n = 8;

// ---

/*
reg [n-1:0]in = 0;
wire [n-1:0]out;

oneMinusA #(3, 4) dut(
    .a(in),
    .b(out)
);
*/



reg [5:0]in = 0;
wire [2:0]out;

sat #(6, 3) dut(
    .a(in),
    .b(out)
);


/*
reg [n*2-1:0]in = 0;
wire [n-1:0]out;

Qmn #(7, 8, 3, 4) dut(
    .a(in),
    .b(out)
);
*/

/*
reg [n-1:0]in = 0;
wire [n-1:0]out;

sigm #(3, 4) dut(
    .a(in),
    .b(out)
);
*/


/*
reg [n-1:0]in = 0;
wire [n-1:0]out;

tanh #(3, 4) dut(
    .a(in),
    .b(out)
);
*/


always begin
    //in = $urandom(); //in+1;
    in = in+1;
    #(1);
end

endmodule
