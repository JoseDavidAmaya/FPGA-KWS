`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/09/2022 09:26:59 PM
// Design Name: 
// Module Name: counter
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


module counter #(parameter n=8)(
    // Inputs
    input clk,
    input rst,
    input en,
    
    input [n-1:0]st_val,
    input [n-1:0]end_val,
    input [n-1:0]add_val,
    
    // Outputs
    output reg [n-1:0]val,
    output valend
    );
    
    wire [n-1:0]valp = val + add_val;

    assign valend = rst ? 0 : valp >= end_val;

    wire [n-1:0]mux = valend ? st_val : valp;
    
    always @(posedge clk) begin
      if (rst)
         val <= st_val;
      else if (en)
         val <= mux;
    end
endmodule
