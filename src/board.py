import numpy as np

class ReversiBoard:
    """Reversi board implementation."""
    
    # Board representation:
    # 0: Empty, 1: Black, -1: White
    # Black goes first
    
    def __init__(self, size=8):
        """Initialize board with the standard starting position."""
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = 1  # Black goes first
        
        # Set up the initial board state
        center = size // 2
        self.board[center-1, center-1] = -1  # White
        self.board[center, center] = -1      # White
        self.board[center-1, center] = 1     # Black
        self.board[center, center-1] = 1     # Black
        
        # Game state
        self.game_over = False
        self.winner = None
    
    def get_valid_moves(self, player=None):
        """Return a list of valid moves for the specified player."""
        if player is None:
            player = self.current_player
            
        valid_moves = []
        
        for i in range(self.size):
            for j in range(self.size):
                if self.is_valid_move(i, j, player):
                    valid_moves.append((i, j))
                    
        return valid_moves
    
    def is_valid_move(self, row, col, player=None):
        """Check if a move is valid for the specified player."""
        if player is None:
            player = self.current_player
            
        # Cell must be empty
        if self.board[row, col] != 0:
            return False
            
        # Must flip at least one opponent's piece
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc, player):
                return True
                
        return False
    
    def _would_flip(self, row, col, dr, dc, player):
        """Check if placing a piece would flip opponent pieces in a direction."""
        opponent = -player
        r, c = row + dr, col + dc
        
        # Check if first piece in this direction is opponent's
        if not (0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == opponent):
            return False
            
        # Keep checking in this direction
        r += dr
        c += dc
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r, c] == 0:
                return False
            if self.board[r, c] == player:
                return True
            r += dr
            c += dc
            
        return False
    
    def make_move(self, row, col, player=None):
        """Make a move on the board, flipping any captured pieces."""
        if player is None:
            player = self.current_player
            
        if not self.is_valid_move(row, col, player):
            return False
            
        # Place the piece
        self.board[row, col] = player
        
        # Flip pieces in all directions
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            self._flip_pieces(row, col, dr, dc, player)
            
        # Switch player
        self.current_player = -player
        
        # Check if next player has any valid moves
        if not self.get_valid_moves(-player):
            # If not, switch back to current player
            self.current_player = player
            
            # If current player also has no moves, game is over
            if not self.get_valid_moves(player):
                self.game_over = True
                self._determine_winner()
                
        return True
    
    def _flip_pieces(self, row, col, dr, dc, player):
        """Flip opponent pieces in a direction after a move."""
        if not self._would_flip(row, col, dr, dc, player):
            return
            
        opponent = -player
        r, c = row + dr, col + dc
        
        # Flip all opponent pieces in this direction
        while self.board[r, c] == opponent:
            self.board[r, c] = player
            r += dr
            c += dc
    
    def _determine_winner(self):
        """Determine the winner of the game."""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        
        if black_count > white_count:
            self.winner = 1  # Black wins
        elif white_count > black_count:
            self.winner = -1  # White wins
        else:
            self.winner = 0  # Draw
    
    def get_state(self):
        """Return the current board state."""
        return self.board.copy()
    
    def get_result(self):
        """Return the result of the game."""
        if not self.game_over:
            return None
        return self.winner
    
    def is_game_over(self):
        """Check if the game is over."""
        return self.game_over
    
    def count_pieces(self):
        """Count pieces for both players."""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        return black_count, white_count
    
    def reset(self):
        """Reset the board to the initial state."""
        self.__init__(self.size)
    
    def __str__(self):
        """String representation of the board."""
        symbols = {0: '.', 1: 'B', -1: 'W'}
        result = []
        
        # Column labels
        result.append('  ' + ' '.join([str(i) for i in range(self.size)]))
        
        for i in range(self.size):
            row = [str(i)]  # Row label
            for j in range(self.size):
                row.append(symbols[self.board[i, j]])
            result.append(' '.join(row))
            
        return '\n'.join(result)