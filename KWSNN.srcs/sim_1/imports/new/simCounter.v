`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/14/2022 12:09:53 PM
// Design Name: 
// Module Name: simCounter
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


module simCounter();
reg clk = 0;

// Counter test
wire [7:0]val0;
wire [7:0]val1;
reg rst = 1;
wire dut1en;

counter dut0(
    .clk(clk),
    .st_val(8'd0),
    .end_val(8'd10),
    .val(val0),
    .add_val(1),
    .en(1),
    .rst(rst),
    .valend(dut1en)
);

counter dut1(
    .clk(clk),
    .st_val(8'd10),
    .end_val(8'd50),
    .add_val(8'd10),
    .val(val1),
    .en(dut1en),
    .rst(rst),
    .valend()
);

wire [8:0] secondAddr = val0 + val1;


always begin
    clk = clk + 1; #1;
end

initial begin
    #200; rst=0;
end

endmodule

