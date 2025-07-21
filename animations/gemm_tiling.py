from manim import *

class GEMMTilingAnimation(Scene):
    def construct(self):
        # --- Constants and Configurations ---
        MATRIX_SIZE = 4 # N x N matrices for simplicity
        TILE_SIZE = 2   # Tile size (e.g., 2x2)

        # Colors
        A_COLOR = BLUE_D
        B_COLOR = GREEN_D
        C_COLOR = YELLOW_D
        SHARED_MEM_COLOR = GREY_B
        HIGHLIGHT_COLOR = RED
        TEXT_COLOR = WHITE

        # Positions
        matrix_A_pos = 3 * LEFT + 1 * UP
        matrix_B_pos = 1 * UP
        matrix_C_pos = 3 * RIGHT + 1 * UP
        shared_mem_pos_A = 3 * LEFT + 2 * DOWN
        shared_mem_pos_B = 1 * LEFT + 2 * DOWN
        shared_mem_pos_C_temp = 1 * RIGHT + 2 * DOWN

        # --- Create Matrices ---
        def create_matrix(rows, cols, label):
            matrix = []
            for i in range(rows):
                row_elements = []
                for j in range(cols):
                    row_elements.append(f"{label}_{i}{j}")
                matrix.append(row_elements)
            return Matrix(matrix).scale(0.7)

        mat_A = create_matrix(MATRIX_SIZE, MATRIX_SIZE, "A")
        mat_A.to_corner(UL).shift(matrix_A_pos)
        A_label = MathTex("A").next_to(mat_A, UP).set_color(A_COLOR)

        mat_B = create_matrix(MATRIX_SIZE, MATRIX_SIZE, "B")
        mat_B.to_corner(UR).shift(matrix_B_pos)
        B_label = MathTex("B").next_to(mat_B, UP).set_color(B_COLOR)

        mat_C = create_matrix(MATRIX_SIZE, MATRIX_SIZE, "C")
        mat_C.to_corner(DR).shift(matrix_C_pos)
        C_label = MathTex("C").next_to(mat_C, UP).set_color(C_COLOR)

        # --- Shared Memory Representation ---
        shared_mem_rect = Rectangle(width=6, height=2.5, color=SHARED_MEM_COLOR, fill_opacity=0.2)
        shared_mem_rect.move_to(2 * DOWN)
        shared_mem_label = Text("Shared Memory", font_size=28).next_to(shared_mem_rect, UP, buff=0.2)

        # Placeholders for tiles in shared memory
        shared_tile_A_mobj = Rectangle(width=TILE_SIZE * 0.7 + (TILE_SIZE-1)*0.1, height=TILE_SIZE * 0.7 + (TILE_SIZE-1)*0.1, color=A_COLOR, fill_opacity=0.5)
        shared_tile_A_mobj.move_to(shared_mem_pos_A)
        shared_tile_A_label = Text("Tile A", font_size=20).next_to(shared_tile_A_mobj, UP)

        shared_tile_B_mobj = Rectangle(width=TILE_SIZE * 0.7 + (TILE_SIZE-1)*0.1, height=TILE_SIZE * 0.7 + (TILE_SIZE-1)*0.1, color=B_COLOR, fill_opacity=0.5)
        shared_tile_B_mobj.move_to(shared_mem_pos_B)
        shared_tile_B_label = Text("Tile B", font_size=20).next_to(shared_tile_B_mobj, UP)
        
        # Temporary C tile in shared memory (for accumulation)
        shared_tile_C_temp_mobj = Rectangle(width=TILE_SIZE * 0.7 + (TILE_SIZE-1)*0.1, height=TILE_SIZE * 0.7 + (TILE_SIZE-1)*0.1, color=C_COLOR, fill_opacity=0.5, stroke_width=2)
        shared_tile_C_temp_mobj.move_to(shared_mem_pos_C_temp)
        shared_tile_C_temp_label = Text("Tile C (temp)", font_size=20).next_to(shared_tile_C_temp_mobj, UP)


        # --- Introduction ---
        title = Text("CUDA Memory Tiling: GEMM Example").to_edge(UP)
        self.play(Create(title))
        self.wait(0.5)

        self.play(
            FadeIn(mat_A, shift=UP),
            FadeIn(A_label, shift=UP),
            FadeIn(mat_B, shift=UP),
            FadeIn(B_label, shift=UP),
            FadeIn(mat_C, shift=UP),
            FadeIn(C_label, shift=UP)
        )
        self.wait(1)

        global_mem_text = Text("Global Memory", font_size=24).next_to(mat_A, LEFT).align_to(mat_B, LEFT).shift(1.5*LEFT)
        self.play(Write(global_mem_text))
        self.wait(1)

        self.play(
            Create(shared_mem_rect),
            Write(shared_mem_label)
        )
        self.wait(1)

        explanation1 = Text("Problem: Large matrices don't fit in fast memory.", font_size=28).to_edge(DOWN)
        self.play(Write(explanation1))
        self.wait(2)
        self.play(FadeOut(explanation1))

        explanation2 = Text("Solution: Break matrices into smaller 'tiles'.", font_size=28).to_edge(DOWN)
        self.play(Write(explanation2))
        self.wait(2)
        self.play(FadeOut(explanation2))

        # --- Tiling Process ---
        def get_tile_indices(matrix_idx, tile_row, tile_col, tile_size):
            start_row = tile_row * tile_size
            end_row = start_row + tile_size - 1
            start_col = tile_col * tile_size
            end_col = start_col + tile_size - 1
            indices = []
            for r in range(start_row, end_row + 1):
                for c in range(start_col, end_col + 1):
                    indices.append((r, c))
            return indices

        def get_submatrix_mobject(matrix_mobj, row_start, col_start, tile_size):
            entries = matrix_mobj.get_entries()
            rows, cols = MATRIX_SIZE, MATRIX_SIZE # Assuming square matrices for simplicity
            
            sub_mobjects = VGroup()
            for r in range(row_start, row_start + tile_size):
                for c in range(col_start, col_start + tile_size):
                    # Manim's Matrix entries are ordered row by row
                    index_in_entries = r * cols + c
                    if index_in_entries < len(entries):
                        sub_mobjects.add(entries[index_in_entries])
            
            if len(sub_mobjects) > 0:
                # Create a rectangle around the selected elements for highlighting
                return SurroundingRectangle(sub_mobjects, color=HIGHLIGHT_COLOR, buff=0.05)
            return None


        num_tiles_per_dim = MATRIX_SIZE // TILE_SIZE

        for tile_row_A in range(num_tiles_per_dim):
            for tile_col_B in range(num_tiles_per_dim):
                # We are calculating C[tile_row_A][tile_col_B]
                target_C_tile_rect = get_submatrix_mobject(mat_C, tile_row_A * TILE_SIZE, tile_col_B * TILE_SIZE, TILE_SIZE)
                self.play(Create(target_C_tile_rect))
                self.wait(0.5)

                for k in range(num_tiles_per_dim):
                    # Highlight tiles in A and B
                    current_tile_A_rect = get_submatrix_mobject(mat_A, tile_row_A * TILE_SIZE, k * TILE_SIZE, TILE_SIZE)
                    current_tile_B_rect = get_submatrix_mobject(mat_B, k * TILE_SIZE, tile_col_B * TILE_SIZE, TILE_SIZE)

                    self.play(
                        Create(current_tile_A_rect),
                        Create(current_tile_B_rect)
                    )
                    self.wait(0.5)

                    # Move tiles to shared memory
                    self.play(
                        Transform(current_tile_A_rect, shared_tile_A_mobj),
                        Transform(current_tile_B_rect, shared_tile_B_mobj),
                        FadeIn(shared_tile_A_label),
                        FadeIn(shared_tile_B_label)
                    )
                    self.wait(0.5)

                    sync_threads_text = Text("Barrier: __syncthreads__()", font_size=20, color=ORANGE).next_to(shared_mem_rect, DOWN)
                    self.play(Write(sync_threads_text))
                    self.wait(0.5)
                    self.play(FadeOut(sync_threads_text))

                    # Simulate computation
                    compute_text = Text(f"Compute {A_COLOR.hex} Tile A * {B_COLOR.hex} Tile B", font_size=24).move_to(shared_mem_rect.get_center())
                    compute_text.set_color_by_tex_to_color_map({
                        "Tile A": A_COLOR,
                        "Tile B": B_COLOR
                    })
                    self.play(Write(compute_text))
                    self.play(FadeIn(shared_tile_C_temp_mobj), FadeIn(shared_tile_C_temp_label))
                    self.wait(1)
                    self.play(FadeOut(compute_text))

                    # Indicate accumulation
                    if k > 0:
                        accumulate_text = Text("Accumulate results in temp C tile", font_size=24).move_to(shared_mem_rect.get_center())
                        self.play(Write(accumulate_text))
                        self.wait(1)
                        self.play(FadeOut(accumulate_text))
                    
                    self.play(
                        FadeOut(shared_tile_A_mobj), FadeOut(shared_tile_A_label),
                        FadeOut(shared_tile_B_mobj), FadeOut(shared_tile_B_label)
                    )
                    self.wait(0.5)

                # Write final C tile from shared memory to global memory
                write_back_text = Text("Write Tiled C from Shared Memory to Global Memory", font_size=24).move_to(shared_mem_rect.get_center())
                self.play(Write(write_back_text))
                self.wait(0.5)

                # Simulate writing the C tile
                temp_c_tile_copy = shared_tile_C_temp_mobj.copy().set_color(C_COLOR).set_fill(opacity=1.0)
                self.play(Transform(temp_c_tile_copy, target_C_tile_rect))
                
                self.play(
                    FadeOut(write_back_text),
                    FadeOut(shared_tile_C_temp_mobj),
                    FadeOut(shared_tile_C_temp_label)
                )
                self.remove(target_C_tile_rect) # Remove the highlight rectangle

                self.wait(1)

        # --- Conclusion ---
        final_text = Text("Memory Tiling: Reduced Global Memory Access, Improved Performance!", font_size=36, color=GREEN).to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2)
        self.play(FadeOut(final_text), FadeOut(title))

        self.wait(2)