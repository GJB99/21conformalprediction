import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import json

# --------------------------------------------------------------------------------
# Card and Deck
# --------------------------------------------------------------------------------
class Card:
    def __init__(self, suit: str, value: str):
        self.suit = suit
        self.value = value
        self.numeric_value = self._get_numeric_value()
    
    def _get_numeric_value(self) -> int:
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 11
        return int(self.value)

    def __str__(self):
        return f"{self.value}-of-{self.suit}"

class Deck:
    def __init__(self, num_decks: int = 6):
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [
            Card(suit, value)
            for _ in range(self.num_decks)
            for suit in suits
            for value in values
        ]
        np.random.shuffle(self.cards)

# --------------------------------------------------------------------------------
# GameState
# --------------------------------------------------------------------------------
class GameState:
    def __init__(self, num_decks: int = 6):
        self.deck = Deck(num_decks=num_decks)
        self.player_hand: List[Card] = []
        self.dealer_hand: List[Card] = []
        self.running_count = 0
        self.outcome: Optional[float] = None  # +1 (win), -1 (loss), 0 (push)

    def _calculate_hand_value(self, hand: List[Card]) -> Tuple[int, bool]:
        value = 0
        aces = 0
        for card in hand:
            if card.value == 'A':
                aces += 1
            else:
                value += card.numeric_value
        is_soft = False
        for _ in range(aces):
            if value + 11 <= 21:
                value += 11
                is_soft = True
            else:
                value += 1
        return value, is_soft

    def get_player_value_and_soft(self) -> Tuple[int, bool]:
        return self._calculate_hand_value(self.player_hand)

    def get_dealer_value_and_soft(self) -> Tuple[int, bool]:
        return self._calculate_hand_value(self.dealer_hand)

    def get_state_features(self) -> Dict:
        player_val, is_soft = self.get_player_value_and_soft()
        dealer_val, _ = self.get_dealer_value_and_soft()
        return {
            'player_sum': player_val,
            'dealer_up_card': dealer_val if self.dealer_hand else 0,
            'running_count': self.running_count,
            'true_count': self.running_count / (len(self.deck.cards) / 52.0),
            'has_ace': any(card.value == 'A' for card in self.player_hand),
            'is_soft': is_soft
        }

# --------------------------------------------------------------------------------
# BlackjackGame
# --------------------------------------------------------------------------------
class BlackjackGame:
    def __init__(self, num_decks: int = 6, hit_soft_17: bool = False):
        self.game_state = GameState(num_decks=num_decks)
        self.hit_soft_17 = hit_soft_17

    def deal_initial_cards(self):
        self.game_state = GameState(num_decks=self.game_state.deck.num_decks)
        for _ in range(2):
            self._deal_card_to_player()
            self._deal_card_to_dealer()
        
        p_val, _ = self.game_state.get_player_value_and_soft()
        d_val, _ = self.game_state.get_dealer_value_and_soft()

        # Immediate blackjacks
        if p_val == 21 and d_val == 21:
            self.game_state.outcome = 0.0
        elif p_val == 21:
            self.game_state.outcome = 1.0
        elif d_val == 21:
            self.game_state.outcome = -1.0

    def hit(self) -> bool:
        self._deal_card_to_player()
        p_val, _ = self.game_state.get_player_value_and_soft()
        return p_val > 21

    def stand(self) -> float:
        if self.game_state.outcome is not None:
            return self.game_state.outcome
        self._dealer_play()
        p_val, _ = self.game_state.get_player_value_and_soft()
        d_val, _ = self.game_state.get_dealer_value_and_soft()
        if d_val > 21:
            self.game_state.outcome = 1.0
        elif p_val > d_val:
            self.game_state.outcome = 1.0
        elif p_val < d_val:
            self.game_state.outcome = -1.0
        else:
            self.game_state.outcome = 0.0
        return self.game_state.outcome

    def _dealer_play(self):
        while True:
            d_val, d_soft = self.game_state.get_dealer_value_and_soft()
            if d_val < 17:
                self._deal_card_to_dealer()
            elif d_val == 17 and d_soft and self.hit_soft_17:
                self._deal_card_to_dealer()
            else:
                break

    def _deal_card_to_player(self):
        c = self.game_state.deck.cards.pop()
        self.game_state.player_hand.append(c)
        self._update_running_count(c)

    def _deal_card_to_dealer(self):
        c = self.game_state.deck.cards.pop()
        self.game_state.dealer_hand.append(c)
        self._update_running_count(c)

    def _update_running_count(self, card: Card):
        if card.numeric_value >= 10:
            self.game_state.running_count -= 1
        elif card.numeric_value <= 6:
            self.game_state.running_count += 1

# --------------------------------------------------------------------------------
# DataCollector
# --------------------------------------------------------------------------------
class DataCollector:
    """
    Collect data ONLY for player sums 12..16.
    Skips data for <12 or >=17 (though it finishes those games internally).
    """
    def __init__(self, num_decks: int = 6, hit_soft_17: bool = False):
        self.game = BlackjackGame(num_decks=num_decks, hit_soft_17=hit_soft_17)
        self.data: List[Dict] = []

    def simulate_games(self, num_games: int = 2000):
        for _ in range(num_games):
            self.game.deal_initial_cards()
            if self.game.game_state.outcome is not None:
                # e.g. immediate blackjack
                self._record_final_game_state("none")
                continue

            p_val, _ = self.game.game_state.get_player_value_and_soft()
            if p_val < 12:
                self._force_hits_below_12()
                continue
            elif p_val >= 17:
                self._force_stand_above_16()
                continue

            # Now we know p_val is 12..16
            self._simulate_hit_once_and_record()
            self._simulate_stand_immediately_and_record()

    def _force_hits_below_12(self):
        while True:
            val, _ = self.game.game_state.get_player_value_and_soft()
            if val >= 12:
                break
            busted = self.game.hit()
            if busted:
                self.game.stand()
                break

    def _force_stand_above_16(self):
        self.game.stand()

    def _simulate_hit_once_and_record(self):
        game_copy = self._copy_game()
        busted = game_copy.hit()
        outcome = game_copy.stand()
        p_val, _ = game_copy.game_state.get_player_value_and_soft()
        d_val, _ = game_copy.game_state.get_dealer_value_and_soft()

        player_cards_str = [str(c) for c in game_copy.game_state.player_hand]
        dealer_cards_str = [str(c) for c in game_copy.game_state.dealer_hand]
        initial_state = self.game.game_state.get_state_features()

        self.data.append({
            **initial_state,
            'action': 'hit',
            'outcome': outcome,
            'player_cards': player_cards_str,
            'dealer_cards': dealer_cards_str,
            'final_player_value': p_val,
            'final_dealer_value': d_val
        })

    def _simulate_stand_immediately_and_record(self):
        game_copy = self._copy_game()
        outcome = game_copy.stand()
        p_val, _ = game_copy.game_state.get_player_value_and_soft()
        d_val, _ = game_copy.game_state.get_dealer_value_and_soft()

        player_cards_str = [str(c) for c in game_copy.game_state.player_hand]
        dealer_cards_str = [str(c) for c in game_copy.game_state.dealer_hand]
        initial_state = self.game.game_state.get_state_features()

        self.data.append({
            **initial_state,
            'action': 'stand',
            'outcome': outcome,
            'player_cards': player_cards_str,
            'dealer_cards': dealer_cards_str,
            'final_player_value': p_val,
            'final_dealer_value': d_val
        })

    def _record_final_game_state(self, action: str):
        p_val, _ = self.game.game_state.get_player_value_and_soft()
        d_val, _ = self.game.game_state.get_dealer_value_and_soft()
        player_cards_str = [str(c) for c in self.game.game_state.player_hand]
        dealer_cards_str = [str(c) for c in self.game.game_state.dealer_hand]
        features = self.game.game_state.get_state_features()
        self.data.append({
            **features,
            'action': action,
            'outcome': self.game.game_state.outcome,
            'player_cards': player_cards_str,
            'dealer_cards': dealer_cards_str,
            'final_player_value': p_val,
            'final_dealer_value': d_val
        })

    def _copy_game(self):
        g = BlackjackGame(num_decks=self.game.game_state.deck.num_decks,
                          hit_soft_17=self.game.hit_soft_17)
        g.game_state.deck.cards = self.game.game_state.deck.cards.copy()
        g.game_state.player_hand = self.game.game_state.player_hand.copy()
        g.game_state.dealer_hand = self.game.game_state.dealer_hand.copy()
        g.game_state.running_count = self.game.game_state.running_count
        g.game_state.outcome = self.game.game_state.outcome
        return g

    def get_training_data(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

# --------------------------------------------------------------------------------
# Conformal Predictor
# --------------------------------------------------------------------------------
class InductiveConformalPredictor:
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibration_scores = []
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        predictions = self.base_model.predict_proba(X_cal)
        for pred, true_label in zip(predictions, y_cal):
            label = 1.0 if true_label == 1.0 else 0.0
            score = abs(pred[1] - label)
            self.calibration_scores.append(score)
        self.calibration_scores.sort()

    def predict(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[List[float], List[float]]:
        predictions = self.base_model.predict_proba(X)
        alpha = 1 - confidence
        # e.g. if alpha=0.05 => quantile=95th percentile of calibration scores
        quantile = np.percentile(self.calibration_scores, alpha * 100)

        lower_list = []
        upper_list = []
        for pred in predictions:
            p_win = pred[1]
            # Lower = p_win - quantile
            # Upper = p_win + quantile
            # clipped to [0,1]
            lower_list.append(max(0.0, p_win - quantile))
            upper_list.append(min(1.0, p_win + quantile))
        return lower_list, upper_list

# --------------------------------------------------------------------------------
# BlackjackPredictor
# --------------------------------------------------------------------------------
class BlackjackPredictor:
    def __init__(self):
        self.base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.conformal_predictor = None
        self.feature_cols = [
            'player_sum',
            'dealer_up_card',
            'running_count',
            'true_count',
            'has_ace',
            'is_soft'
        ]
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df_filtered = df[(df['player_sum'] >= 12) & (df['player_sum'] <= 16)].copy()
        X = df_filtered[self.feature_cols]
        y = df_filtered['outcome'].apply(lambda x: 1.0 if x == 1.0 else 0.0)
        return X, y
    
    def train(self, data: pd.DataFrame, calibration_split: float = 0.2):
        X, y = self.prepare_data(data)
        if X.empty:
            print("Warning: No data for sums 12..16. The model will be empty!")
            return
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=calibration_split, random_state=42
        )
        self.base_model.fit(X_train, y_train)
        self.conformal_predictor = InductiveConformalPredictor(self.base_model)
        self.conformal_predictor.fit(X_cal, y_cal)
    
    def predict(self, state: Dict, confidence: float = 0.95) -> Dict:
        row = {col: state[col] for col in self.feature_cols}
        X = pd.DataFrame([row])
        base_pred = self.base_model.predict_proba(X)[0]
        lower, upper = [0.0], [1.0]
        if self.conformal_predictor is not None:
            low_list, up_list = self.conformal_predictor.predict(X, confidence)
            lower, upper = low_list, up_list
        return {
            'win_probability': base_pred[1],
            'confidence_interval': (lower[0], upper[0])
        }

# --------------------------------------------------------------------------------
# Additional Analysis
# --------------------------------------------------------------------------------
def create_individual_combo_analysis(df: pd.DataFrame):
    """
    For each player_sum in [12..16], produce:
      results/combo_analysis_12.csv
      ...
      results/combo_analysis_16.csv
    Each includes (player_cards, dealer_cards) combos and summary stats.
    """
    df['player_cards'] = df['player_cards'].apply(tuple)
    df['dealer_cards'] = df['dealer_cards'].apply(tuple)
    df['win_bool'] = (df['outcome'] == 1.0).astype(int)

    for psum in range(12, 17):
        df_sub = df[df['player_sum'] == psum].copy()
        if df_sub.empty:
            continue

        summary = df_sub.groupby(['player_cards', 'dealer_cards']).agg(
            n_plays=('outcome', 'count'),
            avg_win=('win_bool', 'mean'),
            avg_player_val=('final_player_value', 'mean'),
            avg_dealer_val=('final_dealer_value', 'mean')
        ).reset_index()

        # Convert tuples back to lists for CSV
        summary['player_cards'] = summary['player_cards'].apply(list)
        summary['dealer_cards'] = summary['dealer_cards'].apply(list)

        outpath = f"results/combo_analysis_{psum}.csv"
        summary.to_csv(outpath, index=False)


def create_decision_matrix_12_16_with_ci(df: pd.DataFrame):
    """
    Create separate CSVs for each player_sum in [12..16], 
    listing (dealer_up_card, recommended_action, confidence_lower, confidence_upper).
    """
    # Filter for sums in [12..16]
    sub = df[(df['player_sum'] >= 12) & (df['player_sum'] <= 16)].copy()

    for psum in range(12, 17):
        tmp = sub[sub['player_sum'] == psum].copy()
        if tmp.empty:
            continue
        # We'll just keep the columns that show the relevant info
        # for each up-card / recommended action, plus the intervals
        # We can drop duplicates if needed
        tmp = tmp[['dealer_up_card', 'recommended_action', 
                   'confidence_lower', 'confidence_upper']].drop_duplicates()

        # Save as e.g. results/decision_matrix_12_with_ci.csv
        outname = f"results/decision_matrix_{psum}_with_ci.csv"
        tmp.to_csv(outname, index=False)

# --------------------------------------------------------------------------------
# Main training & analysis
# --------------------------------------------------------------------------------
def train_model(num_games: int = 50000) -> BlackjackPredictor:
    collector = DataCollector(num_decks=6, hit_soft_17=False)
    collector.simulate_games(num_games=num_games)
    df = collector.get_training_data()

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/game_log.csv", index=False)

    # Create separate combo analyses for sums 12..16
    create_individual_combo_analysis(df)

    predictor = BlackjackPredictor()
    predictor.train(df, calibration_split=0.2)

    os.makedirs("models", exist_ok=True)
    joblib.dump(predictor, "models/blackjack_predictor.pkl")
    return predictor

def generate_all_possibilities():
    possibilities = []
    for player_sum in range(4, 22):
        for dealer_card in range(2, 12):
            for running_count in range(-20, 21):
                true_count = running_count / 6.0
                for has_ace in [True, False]:
                    is_soft = (has_ace and player_sum < 12)
                    possibilities.append({
                        'player_sum': player_sum,
                        'dealer_up_card': dealer_card,
                        'running_count': running_count,
                        'true_count': true_count,
                        'has_ace': has_ace,
                        'is_soft': is_soft,
                    })
    return possibilities

def analyze_and_save_results(predictor: BlackjackPredictor):
    possibilities = generate_all_possibilities()
    results = []
    
    for state in possibilities:
        p_sum = state['player_sum']
        
        # Hard-coded logic for <12 => hit, >=17 => stand
        if p_sum < 12:
            recommended_action = 'hit'
            results.append({
                **state,
                'win_probability': 0.0,
                'confidence_lower': 0.0,
                'confidence_upper': 1.0,
                'recommended_action': recommended_action
            })
            continue
        elif p_sum >= 17:
            recommended_action = 'stand'
            results.append({
                **state,
                'win_probability': 0.0,
                'confidence_lower': 0.0,
                'confidence_upper': 1.0,
                'recommended_action': recommended_action
            })
            continue
        
        # If 12..16, consult the model
        pred = predictor.predict(state, confidence=0.95)
        low_ci, up_ci = pred['confidence_interval']
        mid_prob = pred['win_probability']
        
        # Example interval-based approach
        if low_ci > 0.5:
            recommended_action = 'stand'
        elif up_ci < 0.3:
            recommended_action = 'hit'
        else:
            recommended_action = 'stand' if mid_prob > 0.5 else 'hit'
        
        results.append({
            **state,
            'win_probability': float(mid_prob),
            'confidence_lower': float(low_ci),
            'confidence_upper': float(up_ci),
            'recommended_action': recommended_action
        })

    df = pd.DataFrame(results)
    df.to_csv("results/full_analysis.csv", index=False)

    # Create a pivot table for (player_sum, dealer_up_card) with recommended actions
    zero_count_df = df[(df['running_count'] == 0) & (df['true_count'] == 0)]
    decision_matrix = zero_count_df.pivot_table(
        index='player_sum',
        columns='dealer_up_card',
        values='recommended_action',
        aggfunc='first'
    )
    decision_matrix.to_csv("results/decision_matrix.csv")

    # Also create separate CSVs for player_sum=12..16 that show
    # (dealer_up_card, recommended_action, confidence_lower, confidence_upper).
    create_decision_matrix_12_16_with_ci(df)

    summary = {
        'total_scenarios': len(df),
        'hit_percentage': float((df['recommended_action'] == 'hit').mean() * 100),
        'stand_percentage': float((df['recommended_action'] == 'stand').mean() * 100),
    }
    with open("results/summary.json", "w") as fp:
        json.dump(summary, fp, indent=4)

def main():
    predictor = train_model(num_games=1000000)
    analyze_and_save_results(predictor)
    print("Analysis complete!")
    print(" - 'results/game_log.csv': all final outcomes (cards, sums, outcome).")
    print(" - 'results/combo_analysis_12.csv'..'combo_analysis_16.csv': combos for each sum in [12..16].")
    print(" - 'results/full_analysis.csv': final strategy table for [4..21], incl. intervals.")
    print(" - 'results/decision_matrix.csv': pivoted matrix for zero-count scenario.")
    print(" - 'results/decision_matrix_{sum}_with_ci.csv': for sums 12..16, includes confidence intervals.")

if __name__ == "__main__":
    main()
