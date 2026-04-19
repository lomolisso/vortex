module sram_256x24_1r1w
(
   rd_out,
   r_addr_in,
   r_ce_in,
   w_addr_in,
   w_ce_in,
   wd_in,
   w_mask_in,
   clk
);
   parameter BITS = 24;
   parameter WORD_DEPTH = 256;
   parameter ADDR_WIDTH = 8;
   parameter corrupt_mem_on_X_p = 1;

   output reg [BITS-1:0]    rd_out;
   input  [ADDR_WIDTH-1:0]  r_addr_in;
   input                    r_ce_in;
   input  [ADDR_WIDTH-1:0]  w_addr_in;
   input                    w_ce_in;
   input  [BITS-1:0]        wd_in;
   input  [BITS-1:0]        w_mask_in;
   input                    clk;

   reg    [BITS-1:0]        mem [0:WORD_DEPTH-1];

   integer j;

   always @(posedge clk)
   begin

      // Write port
      if (w_ce_in)
      begin
         if (corrupt_mem_on_X_p &&
             ((^w_ce_in === 1'bx) || (^w_addr_in === 1'bx))
            )
         begin
            for (j = 0; j < WORD_DEPTH; j = j + 1)
               mem[j] <= 'x;
            $display("warning: w_ce_in or w_addr_in is unknown in sram_256x24_1r1w");
         end
         else
         begin
            mem[w_addr_in] <= (wd_in & w_mask_in) | (mem[w_addr_in] & ~w_mask_in);
         end
      end

      // Read port - READ-FIRST: rd_out always reflects the array state at the
      // start of the cycle, even when r_addr_in == w_addr_in.
      if (r_ce_in)
      begin
         if (corrupt_mem_on_X_p && (^r_addr_in === 1'bx))
         begin
            rd_out <= 'x;
            $display("warning: r_addr_in is unknown in sram_256x24_1r1w");
         end
         else
         begin
            rd_out <= mem[r_addr_in];
         end
      end
      else
      begin
         rd_out <= 'x;
      end

   end
   // Timing check placeholders (will be replaced during SDF back-annotation)
   reg notifier;
   specify
      // Clock-to-Q delay
      (posedge clk *> rd_out) = (0, 0);

      // Timing checks
      $width     (posedge clk,               0, 0, notifier);
      $width     (negedge clk,               0, 0, notifier);
      $period    (posedge clk,               0,    notifier);
      $setuphold (posedge clk,    r_ce_in, 0, 0, notifier);
      $setuphold (posedge clk,    w_ce_in, 0, 0, notifier);
      $setuphold (posedge clk,  r_addr_in, 0, 0, notifier);
      $setuphold (posedge clk,  w_addr_in, 0, 0, notifier);
      $setuphold (posedge clk,      wd_in, 0, 0, notifier);
      $setuphold (posedge clk,  w_mask_in, 0, 0, notifier);
   endspecify

endmodule
