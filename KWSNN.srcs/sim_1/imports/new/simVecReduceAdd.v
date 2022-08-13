`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/08/2022 07:25:30 AM
// Design Name: 
// Module Name: simVecReduceAdd
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


module simVecReduceAdd();

reg clk = 0;

localparam n = 16;
localparam nn = n+1;
localparam m = 8;
genvar i;

wire [n-1:0]a[m-1:0];

reg [n*m-1:0]aflat;
for (i = 0; i < m; i=i+1) begin assign a[i] = aflat[(i+1)*n-1:i*n]; end // unflatten array

wire [nn-1:0]out;

vecReduceAddLLseg #(n, m, nn) dut(
//vecReduceAddLL #(n, m, nn) dut(
    .clk(clk),
    .vecA(aflat),
    .vecB(out)
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
