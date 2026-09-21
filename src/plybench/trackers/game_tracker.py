from __future__ import annotations

from typing import Any

from plybench.common.enums import GameResults
from plybench.common.serializable import Serializable
from plybench.configs.parser import ConfigParser
from plybench.configs.player_config import PlayerConfig
from plybench.core.interface import InterfaceObservation
from plybench.player.player import PlayerOutput
from plybench.trackers.player_tracker import PlayerTrackerResolver


class GameEnding(Serializable):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameEnding:
        result = GameResults.from_value(data["result"])
        if result is None:
            raise ValueError(f"Invalid game result: {data['result']}")
        return cls(data["seq"], data["observation"], result)

    def __init__(self, seq: int, observation: str, result: GameResults) -> None:
        self.seq = seq
        self.observation = observation
        self.result = result

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "observation": self.observation, "result": self.result.value}


class GameStep(Serializable):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameStep:
        return cls(
            data["seq"],
            data["player_name"],
            data["player_hash"],
            data["serialized_state"],
            data["observation"],
            data["move"],
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            reasoning_tokens=data.get("reasoning_tokens"),
            data=data.get("data", None),
        )

    def __init__(
        self,
        seq: int,
        player_name: str,
        player_hash: str,
        serialized_state: str,
        observation: str,
        move: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        # core benchmark-trackable fields
        self.seq = seq
        self.player_name = player_name
        self.player_hash = player_hash
        self.serialized_state = serialized_state
        self.observation = observation
        self.move = move
        # per-turn token usage; None means this player type has no token concept (e.g. a bot),
        # which is distinct from a genuine 0 (a model call that produced zero of that kind)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        # player-specific extras produced by the registered PlayerTracker
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "seq": self.seq,
            "player_name": self.player_name,
            "player_hash": self.player_hash,
            "serialized_state": self.serialized_state,
            "observation": self.observation,
            "move": self.move,
        }
        if self.input_tokens is not None:
            res["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            res["output_tokens"] = self.output_tokens
        if self.reasoning_tokens is not None:
            res["reasoning_tokens"] = self.reasoning_tokens
        if self.data:
            res["data"] = self.data
        return res


class GameTracker(Serializable[[ConfigParser]]):
    @classmethod
    def from_dict(cls, data: dict[str, Any], parser: ConfigParser) -> GameTracker:
        return cls(
            int(data["game_round"]),
            parser.player_config(data["i_player"]),
            parser.player_config(data["o_player"]),
            data["instance_params"],
            [GameStep.from_dict(step) for step in data["steps"]] if data["steps"] else [],
            GameEnding.from_dict(data["ending"]) if data["ending"] else None,
            int(data["seq"]),
            data.get("other_params", {}),
        )

    def __init__(
        self,
        game_round: int,
        i_player: PlayerConfig,
        o_player: PlayerConfig,
        instance_params: dict[str, Any],
        steps: list[GameStep] | None = None,
        ending: GameEnding | None = None,
        seq: int = 0,
        other_params: dict[str, Any] | None = None,
    ) -> None:
        self.game_round = game_round
        self.seq = seq
        self.i_player = i_player
        self.o_player = o_player
        self.instance_params = instance_params
        self.other_params = other_params
        self.steps: list[GameStep] = steps if steps is not None else []
        self.ending: GameEnding | None = ending

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def steps_of(self, player: PlayerConfig, attr: str | None = None) -> list[GameStep]:
        # `attr` drops steps whose field is None, i.e. the player type has no such concept (a bot
        # records no tokens), which is distinct from a genuine 0
        return [step for step in self.steps if step.player_hash == player.hash and (attr is None or getattr(step, attr) is not None)]

    def add_move(
        self, player: PlayerConfig, observation: InterfaceObservation, player_output: PlayerOutput, serialized_state: str, trackers: PlayerTrackerResolver | None = None
    ) -> None:
        if player_output.action is None:
            move = f"FAIL: {player_output.failure_reason}"
        else:
            move = player_output.action.to_llm().string

        # tokens are benchmark-trackable and recorded generically from the player output;
        # the player's registered tracker (if any) contributes the player-specific `data` extras.
        data = trackers.player_tracker(player.key).record(player_output) if trackers is not None else {}

        self.steps.append(
            GameStep(
                self._next_seq(),
                player.to_string(),  # player_name = canonical serialization string
                player.hash,  # short hash for matching steps to a player
                serialized_state,
                str(observation),
                move,
                input_tokens=player_output.input_tokens,
                output_tokens=player_output.output_tokens,
                reasoning_tokens=player_output.reasoning_tokens,
                data=data or None,
            )
        )

    def add_fail(self, player: PlayerConfig, observation: InterfaceObservation) -> None:
        if player.hash == self.i_player.hash:
            self.end_game(GameResults.MY_FAIL, observation)
        elif player.hash == self.o_player.hash:
            self.end_game(GameResults.OPPONENT_FAIL, observation)
        else:
            raise ValueError(f"Invalid player hash: {player.hash}")

    def end_game(self, result: GameResults, observation: InterfaceObservation) -> None:
        self.ending = GameEnding(self._next_seq(), str(observation), result)

    def get_result(self, player: PlayerConfig) -> GameResults:
        assert self.ending is not None, "Game must be ended to get the result"
        if player.hash == self.i_player.hash:
            return self.ending.result
        return self.ending.result.invert()

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_round": self.game_round,
            "i_player": self.i_player.to_string(),
            "o_player": self.o_player.to_string(),
            "instance_params": self.instance_params,
            "other_params": self.other_params if self.other_params else {},
            "steps": [step.to_dict() for step in self.steps],
            "ending": self.ending.to_dict() if self.ending else None,
            "seq": self.seq,
        }
