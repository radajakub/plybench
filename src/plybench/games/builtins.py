from plybench.configs.game_params import NoGameParams
from plybench.games.bandit.bandit import BanditEngine, BanditGameParams
from plybench.games.breakthrough.breakthrough import BreakthroughEngine
from plybench.games.connect_four.connect_four import ConnectFourEngine
from plybench.games.monty_hall.monty_hall import MontyHallEngine, MontyHallGameParams
from plybench.games.nim.inverse_nim import InverseNimEngine
from plybench.games.nim.modified_nim import ModifiedNimEngine
from plybench.games.nim.nim import NimEngine, NimGameParams
from plybench.games.nim.story_nim import StoryNimEngine
from plybench.games.spec import GameSpec
from plybench.games.tic_tac_toe.magic_square import MagicSquareEngine, MagicSquareGameParams
from plybench.games.tic_tac_toe.modified_tic_tac_toe import ModifiedTicTacToeEngine
from plybench.games.tic_tac_toe.story_magic_square import StoryMagicSquareEngine
from plybench.games.tic_tac_toe.tic_tac_toe import TicTacToeEngine
from plybench.registry import Registry

# Built-in game specs. External games register their own GameSpec into op.registry the same way.
_BUILTIN_GAMES: list[GameSpec] = [
    # tic tac toe games
    GameSpec("tic_tac_toe", NoGameParams, TicTacToeEngine, solvable=True),
    GameSpec("modified_tic_tac_toe", NoGameParams, ModifiedTicTacToeEngine, solvable=True),
    GameSpec("magic_square", MagicSquareGameParams, MagicSquareEngine, solvable=True),
    GameSpec("story_magic_square", MagicSquareGameParams, StoryMagicSquareEngine, solvable=True),
    # nim games
    GameSpec("nim", NimGameParams, NimEngine, solvable=True),
    GameSpec("modified_nim", NoGameParams, ModifiedNimEngine, solvable=True),
    GameSpec("inverse_nim", NimGameParams, InverseNimEngine, solvable=True),
    GameSpec("story_nim", NimGameParams, StoryNimEngine, solvable=True),
    # connect 4
    GameSpec("connect_four", NoGameParams, ConnectFourEngine, solvable=False),
    # breakthrough
    GameSpec("breakthrough", NoGameParams, BreakthroughEngine, solvable=False),
    # single-player games
    GameSpec("monty_hall", MontyHallGameParams, MontyHallEngine, solvable=True),
    GameSpec("bandit", BanditGameParams, BanditEngine, solvable=True),
]


def register_builtin_games(registry: Registry) -> None:
    for spec in _BUILTIN_GAMES:
        registry.register_game(spec)
