`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/26/2022 11:37:48 AM
// Design Name: 
// Module Name: simNN
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


module simNN();
`include "NNparams.vh"

genvar i;

reg clk = 0;
reg rst = 1;

localparam t = 5;

always begin
    clk = clk + 1; #(1*t);
end

reg st = 0;
wire rdy;

reg ren_vecout = 0;
reg [addrwidth_TempData1-1:0]r_addr_vecout = 0;
wire[memwidth-1:0] r_data_vecout;

NN dut(
    .clk(clk),
    .st(st),
    .rst(rst),
    
    .access_r_data_temp(r_data_vecout),
    .access_addr_temp(r_addr_vecout),
    .access_en_temp(ren_vecout),

    .access_addr_indata(0),
    .access_w_data_indata(0),
    .access_en_indata(0),
    .access_w_en_indata(0),

    .rdy(rdy)
);

wire [elemwidth-1:0]vecOutArranged[m-1:0];
for (i = 0; i < m; i=i+1) begin assign vecOutArranged[i] = r_data_vecout[i*(elemwidth)+:(elemwidth)]; end // flatten array

// ----- Verification
// Memory containing the result of an operation
reg [memwidth-1:0]pythonRes[249:0];
initial begin
    $readmemh(memOutputInitFileSim, pythonRes);
end

wire [memwidth-1:0]pythonRes_val = pythonRes[r_addr_vecout];

wire RES_COMPARISON = pythonRes_val == r_data_vecout;
// -----

initial begin
    #(t);
    rst = 1; #(t*10000);
    rst = 0; #(t*2);
    st = 1; #(t*2);
    st = 0;
    
    @(posedge rdy);
    #(t*4*5)
    
    ren_vecout = 1; #(t*2);
    r_addr_vecout = 0; #(t*2*10);
    repeat (250) begin
        r_addr_vecout = r_addr_vecout + 1; #(t*2*10);
    end
    ren_vecout = 0; #(t*2);
    r_addr_vecout = 0; #(t*2);
end

endmodule
