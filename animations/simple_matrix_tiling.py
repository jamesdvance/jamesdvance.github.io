from manim import *

class SimpleMatrixTiling(Scene):
    def construct(self):
        # Configuration
        MATRIX_SIZE = 4
        TILE_SIZE = 2
        
        # Colors
        A_COLOR = BLUE
        B_COLOR = GREEN
        C_COLOR = RED
        TILE_COLOR = YELLOW
        
        # Create simple matrix representations
        def create_matrix_grid(size, color, pos):
            squares = VGroup()
            for i in range(size):
                for j in range(size):
                    square = Square(0.5, color=color, fill_opacity=0.3)
                    square.move_to(pos + RIGHT * j * 0.6 + DOWN * i * 0.6)
                    squares.add(square)
            return squares
        
        # Position matrices
        mat_A = create_matrix_grid(MATRIX_SIZE, A_COLOR, 3 * LEFT)
        mat_B = create_matrix_grid(MATRIX_SIZE, B_COLOR, ORIGIN)
        mat_C = create_matrix_grid(MATRIX_SIZE, C_COLOR, 3 * RIGHT)
        
        # Labels
        label_A = Text("Matrix A", font_size=24).next_to(mat_A, UP)
        label_B = Text("Matrix B", font_size=24).next_to(mat_B, UP)
        label_C = Text("Matrix C", font_size=24).next_to(mat_C, UP)
        
        # Title
        title = Text("Simple Matrix Multiplication Tiling", font_size=32).to_edge(UP)
        
        # Show initial setup
        self.play(Write(title))
        self.play(
            Create(mat_A), Write(label_A),
            Create(mat_B), Write(label_B),
            Create(mat_C), Write(label_C)
        )
        self.wait(1)
        
        # Explanation
        explanation = Text("Instead of loading entire matrices,\nwe work with small tiles", 
                         font_size=24).to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))
        
        # Function to highlight a tile in a matrix
        def highlight_tile(matrix, start_row, start_col, tile_size):
            tile_squares = VGroup()
            for i in range(tile_size):
                for j in range(tile_size):
                    idx = (start_row + i) * MATRIX_SIZE + (start_col + j)
                    if idx < len(matrix):
                        tile_squares.add(matrix[idx])
            return SurroundingRectangle(tile_squares, color=TILE_COLOR, buff=0.05)
        
        # Show tiling process
        for tile_row in range(MATRIX_SIZE // TILE_SIZE):
            for tile_col in range(MATRIX_SIZE // TILE_SIZE):
                # Highlight current tile in C
                c_tile_rect = highlight_tile(mat_C, tile_row * TILE_SIZE, tile_col * TILE_SIZE, TILE_SIZE)
                self.play(Create(c_tile_rect))
                
                step_text = Text(f"Computing C tile ({tile_row}, {tile_col})", 
                               font_size=24).to_edge(DOWN)
                self.play(Write(step_text))
                self.wait(0.5)
                
                # Show the tiles needed from A and B
                for k in range(MATRIX_SIZE // TILE_SIZE):
                    # Highlight tiles in A and B
                    a_tile_rect = highlight_tile(mat_A, tile_row * TILE_SIZE, k * TILE_SIZE, TILE_SIZE)
                    b_tile_rect = highlight_tile(mat_B, k * TILE_SIZE, tile_col * TILE_SIZE, TILE_SIZE)
                    
                    self.play(Create(a_tile_rect), Create(b_tile_rect))
                    
                    # Show multiplication
                    mult_text = Text(f"A[{tile_row},{k}] × B[{k},{tile_col}]", 
                                   font_size=20).move_to(UP * 2)
                    self.play(Write(mult_text))
                    self.wait(0.8)
                    
                    self.play(FadeOut(mult_text), FadeOut(a_tile_rect), FadeOut(b_tile_rect))
                
                self.play(FadeOut(step_text), FadeOut(c_tile_rect))
                self.wait(0.3)
        
        # Final message
        final_text = Text("Tiling reduces memory traffic\nand improves cache efficiency!", 
                         font_size=28, color=GREEN).to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(3)
        
        # Cleanup
        self.play(FadeOut(final_text))
        self.wait(1)