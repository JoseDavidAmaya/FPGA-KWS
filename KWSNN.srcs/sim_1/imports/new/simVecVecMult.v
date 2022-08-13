`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 06:13:31 PM
// Design Name: 
// Module Name: simVecVecMult
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


module simVecVecMult();

localparam n = 8;
localparam m = 5;
genvar i;

wire [n-1:0]a[m-1:0];
wire [n-1:0]b[m-1:0];

reg [n*m-1:0]aflat;
reg [n*m-1:0]bflat;
for (i = 0; i < m; i=i+1) begin assign a[i] = aflat[(i+1)*n-1:i*n]; end // unflatten array
for (i = 0; i < m; i=i+1) begin assign b[i] = bflat[(i+1)*n-1:i*n]; end // unflatten array

wire [2*m*n-1:0]out;

wire [2*n-1:0]outVec[m-1:0];
for (i = 0; i < m; i=i+1) begin assign outVec[i] = out[(i+1)*2*n-1:i*2*n]; end // unflatten array

vecVecMult #(n, m) dut(
    .vecA(aflat),
    .vecB(bflat),
    .vecC(out)
);

always begin
    aflat = $urandom + $urandom * 2**8;
    bflat = $urandom + $urandom * 2**8;
    #(1);
end

endmodule
