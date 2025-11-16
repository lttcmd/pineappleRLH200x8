#pragma once

#include "BotPolicy.hpp"
#include "GameState.hpp"
#include <vector>
#include <array>

namespace ofc {

struct BeamElement {
    PlayerBoard board;
    double eval;
};

using Beam = std::vector<BeamElement>;

// Row category classification
enum class RowCategory {
    Empty,
    HighCard,
    LowPair,      // 2-6
    MidPair,      // 7-T
    HighPair,     // J-A
    TwoPair,
    Trips,
    Straight,
    Flush,
    FullHouse,
    QuadsPlus
};

// Row features extracted from analysis
struct RowFeatures {
    RowCategory category;
    int highestRank;          // 2-14
    int pairCount;
    bool tripsPresent;
    bool fourToFlush;
    bool threeToFlush;
    enum StraightPotential { None, Gutshot, OpenEnded };
    StraightPotential straightPotential;
};

// Row evaluation result (made + draw)
struct RowEval {
    double madeScore;
    double drawScore;
};

class BeamSearchPolicy : public BotPolicy {
public:
    BeamSearchPolicy(int beamSize = 20, int rollouts = 10, bool debug = false);
    
    Move chooseInitial(const GameState& state) override;
    Move chooseTurn(const GameState& state) override;

private:
    int beamSize_;
    int rollouts_;  // Number of Monte Carlo rollouts per board evaluation
    bool debug_;
    
    // New structured evaluation system
    double evalBoard(const PlayerBoard& board, int remainingCards) const;
    RowFeatures analyseRow(const std::vector<Card>& cards, const std::string& row) const;
    RowEval evalRow(const std::vector<Card>& cards, const std::string& row, int remainingCards) const;
    double foulRiskPenalty(const PlayerBoard& board, int remainingCards) const;
    
    // Helper functions for row analysis
    int getRankValue(char rank) const;
    int getCategoryRank(RowCategory category) const;
    int calculateRemainingCards(const PlayerBoard& board) const;
    
    // Legacy evaluation methods (kept for fallback)
    double evalBoardStructured(const PlayerBoard& board) const;
    double evalBoardMonteCarlo(const GameState& state, const PlayerBoard& board) const;
    
    // Legacy helper functions (kept for compatibility)
    int getHandCategory(const std::vector<Card>& cards) const; // 0=high card, 1=pair, 2=two pair, 3=trips, 4=straight, 5=flush, 6=full house, 7=quads, 8=straight flush
    int getPairRank(const std::vector<Card>& cards) const; // Returns rank of pair (0 if no pair, A=14)
    bool isLowPair(int rank) const; // Returns true if pair rank is 2-6
    bool isMediumPair(int rank) const; // Returns true if pair rank is 7-T
    bool isHighPair(int rank) const; // Returns true if pair rank is J-A
    int getHighestCard(const std::vector<Card>& cards) const; // Returns highest card rank (A=14, K=13, etc.)
    int getCategoryValue(int category, const std::vector<Card>& cards, const std::string& row) const; // Get base value for a category in a specific row
    
    // Expand initial 5 cards into beam
    Beam expandInitialFive(const GameState& state, const std::vector<Card>& initialCards, int beamSize) const;
    
    // Expand beam with 3 new cards
    Beam expandWithThreeCards(const GameState& state,
                              const Beam& currentBeam, 
                              const std::vector<Card>& newCards, 
                              int beamSize) const;
    
    // Generate sensible initial placements (pruned)
    std::vector<PlayerBoard> generateInitialPlacements(const std::vector<Card>& cards) const;
    
    // Generate sensible turn placements (pruned)
    std::vector<PlayerBoard> generateTurnPlacements(const PlayerBoard& board,
                                                    const std::vector<Card>& newCards) const;
    
    // Helper: find best move to reach target board
    Move boardToMove(const GameState& state, const PlayerBoard& target) const;
};

} // namespace ofc

