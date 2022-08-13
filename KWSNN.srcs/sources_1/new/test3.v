`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/02/2022 01:33:06 AM
// Design Name: 
// Module Name: test3
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


module test3(
        input clk,
        input rst,
        input st,
        
        output rdy,
        
        output [6:0]sevenSegment,
        output [7:0]segment
    );
    
    wire [7:0]idx;
    /*
    NN nnblock(
        .clk(clk),
        .rst(rst),
        .st(st),
        .outIndex(idx),
        .rdy(rdy)
    );
    
    sevenSegLabelShow sevsegshowblock(
        .clk(clk),
        .rst(rst),
        .idx(idx[3:0]),
        
        .sevenSegment(sevenSegment),
        .segment(segment)
    );
    */
    
endmodule
