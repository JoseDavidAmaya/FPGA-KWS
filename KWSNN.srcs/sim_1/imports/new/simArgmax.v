`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/30/2022 05:11:26 PM
// Design Name: 
// Module Name: simArgmax
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


module simArgmax();

reg clk = 0;

localparam n = 8;
localparam m = 8;
genvar i;

wire [n-1:0]a[m-1:0];

reg [n*m-1:0]aflat;
for (i = 0; i < m; i=i+1) begin assign a[i] = aflat[(i+1)*n-1:i*n]; end // unflatten array

wire [n-1:0]out;

vecArgmax #(n, m) dut(
    .vecA(aflat),
    .index(out)
);


always begin
    clk = clk + 1; #1;
end

always begin
    aflat[0+:32] = $urandom;
    aflat[32+:32] = $urandom;
    aflat[64+:32] = $urandom;
    aflat[96+:32] = $urandom;
    #(1);
end

endmodule
