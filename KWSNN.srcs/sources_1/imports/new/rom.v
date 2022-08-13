`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/15/2022 05:41:43 AM
// Design Name: 
// Module Name: rom
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


module rom
    #(
       parameter ROM_WIDTH = 16,
       parameter ROM_ADDR_BITS = 8,
       parameter INIT_FILE = ""
    )
    (
       input clk,
       input r_en,
       input [ROM_ADDR_BITS-1:0] r_addr,
       output [ROM_WIDTH-1:0] r_data
    );
    
   (* rom_style="{distributed | block}" *) reg [ROM_WIDTH-1:0] mem [(2**ROM_ADDR_BITS)-1:0];
   reg [ROM_WIDTH-1:0] ram_data = 0;
   
   initial
      $readmemh(INIT_FILE, mem); //, 0, (2**ROM_ADDR_BITS)-1);
    
   always @(posedge clk)
      if (r_en)
         ram_data <= mem[r_addr];
   
   // Output reg
   reg [ROM_WIDTH-1:0] r_data_reg = 0;
   always @(posedge clk) begin
        r_data_reg <= ram_data;
   end
        
   assign r_data = r_data_reg;
				
endmodule
