#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <set>
#include <vector>

#include "Rules.hpp"

namespace py = pybind11;

static inline int rank_from_int(int c) { return (c / 4) + 2; }   // 2..14
static inline int suit_from_int(int c) { return (c % 4); }       // 0..3

static bool is_straight_5(const std::array<int,5>& ranks_sorted_desc) {
  // Convert to unique sorted ascending
  std::vector<int> tmp(ranks_sorted_desc.begin(), ranks_sorted_desc.end());
  std::sort(tmp.begin(), tmp.end());
  tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
  if (tmp.size() != 5) return false;
  // Wheel
  if (tmp[0]==2 && tmp[1]==3 && tmp[2]==4 && tmp[3]==5 && tmp[4]==14) return true;
  for (int i=0;i<4;++i) if (tmp[i+1]-tmp[i] != 1) return false;
  return true;
}

static std::string evaluate_5(const std::array<int,5>& ranks_desc, const std::array<int,5>& suits) {
  std::array<int,5> ranks = ranks_desc;
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());
  std::unordered_map<int,int> cnt;
  for (int r: ranks) cnt[r]++;
  std::array<int,5> suits_arr = suits;
  bool is_flush = (suits_arr[0]==suits_arr[1] && suits_arr[1]==suits_arr[2] && suits_arr[2]==suits_arr[3] && suits_arr[3]==suits_arr[4]);
  bool is_straight = is_straight_5(ranks);
  std::set<int> rset(ranks.begin(), ranks.end());
  bool is_royal = is_flush && (rset == std::set<int>{14,13,12,11,10});

  if (is_royal) return "royal_flush";
  if (is_straight && is_flush) return "straight_flush";
  // counts
  std::array<int,5> counts{};
  int idx=0;
  for (auto& kv: cnt) counts[idx++] = kv.second;
  std::sort(counts.begin(), counts.begin()+idx, std::greater<int>());
  if (idx>=2 && counts[0]==4) return "quads";
  if (idx>=2 && counts[0]==3 && counts[1]==2) return "full_house";
  if (is_flush) return "flush";
  if (is_straight) return "straight";
  if (counts[0]==3) return "trips";
  if (idx>=3 && counts[0]==2 && counts[1]==2) return "two_pair";
  if (counts[0]==2) return "pair";
  return "high_card";
}

static int top_pair_royalty(const std::array<int,3>& ranks) {
  std::unordered_map<int,int> cnt;
  for (int r: ranks) cnt[r]++;
  for (auto& kv: cnt) {
    if (kv.second==2) {
      // map 6..14
      switch(kv.first){
        case 6: return 1; case 7: return 2; case 8: return 3; case 9: return 4;
        case 10: return 5; case 11: return 6; case 12: return 7; case 13: return 8; case 14: return 9;
        default: return 0;
      }
    }
  }
  return 0;
}

static int top_trips_royalty(const std::array<int,3>& ranks) {
  if (ranks[0]==ranks[1] && ranks[1]==ranks[2]) {
    int r = ranks[0];
    switch(r){
      case 2: return 10; case 3: return 11; case 4: return 12; case 5: return 13; case 6: return 14;
      case 7: return 15; case 8: return 16; case 9: return 17; case 10: return 18; case 11: return 19;
      case 12: return 20; case 13: return 21; case 14: return 22;
      default: return 0;
    }
  }
  return 0;
}

static int royalties_five(const std::array<int,5>& ranks, const std::array<int,5>& suits, bool is_middle){
  std::string t = evaluate_5(ranks, suits);
  if (is_middle){
    if (t=="royal_flush") return 50;
    if (t=="straight_flush") return 30;
    if (t=="quads") return 20;
    if (t=="full_house") return 12;
    if (t=="flush") return 8;
    if (t=="straight") return 4;
    if (t=="trips") return 2;
    return 0;
  } else {
    if (t=="royal_flush") return 25;
    if (t=="straight_flush") return 15;
    if (t=="quads") return 10;
    if (t=="full_house") return 6;
    if (t=="flush") return 4;
    if (t=="straight") return 2;
    return 0;
  }
}

static bool validate_board(const std::array<int,5>& bottom_r, const std::array<int,5>& bottom_s,
                           const std::array<int,5>& middle_r, const std::array<int,5>& middle_s,
                           const std::array<int,3>& top_r){
  // bottom > middle
  auto type_rank = [](const std::string& t)->int{
    if (t=="royal_flush") return 9;
    if (t=="straight_flush") return 8;
    if (t=="quads") return 7;
    if (t=="full_house") return 6;
    if (t=="flush") return 5;
    if (t=="straight") return 4;
    if (t=="trips") return 3;
    if (t=="two_pair") return 2;
    if (t=="pair") return 1;
    return 0;
  };
  std::string bot_t = evaluate_5(bottom_r, bottom_s);
  std::string mid_t = evaluate_5(middle_r, middle_s);
  if (type_rank(bot_t) <= type_rank(mid_t)) return false;
  // middle > top: simple rule approximate (matches Python simplification)
  //
  // IMPORTANT: For ordering we must treat *all* pairs/trips on top as such,
  // even when they do not earn royalties (e.g. 22–55). Royalties are a
  // separate concern.
  std::array<int,15> cnt{};
  cnt.fill(0);
  for (int r : top_r) {
    if (r >= 2 && r <= 14) cnt[r]++;
  }
  bool top_is_trips = false;
  bool top_is_pair = false;
  for (int r = 2; r <= 14; ++r) {
    if (cnt[r] == 3) {
      top_is_trips = true;
      break;
    }
    if (cnt[r] == 2) {
      top_is_pair = true;
    }
  }

  // If top trips, middle must be full house or better or higher trips.
  if (top_is_trips){
    if (!(mid_t=="full_house" || mid_t=="quads" || mid_t=="straight_flush" || mid_t=="royal_flush" || mid_t=="trips")){
      return false;
    }
  } else {
    // If top pair (any rank), middle must be at least trips.
    if (top_is_pair){
      if (mid_t=="high_card" || mid_t=="pair" || mid_t=="two_pair") return false;
    } else {
      // top high card: require middle high card higher top high
      if (mid_t=="high_card"){
        int top_max = std::max({top_r[0], top_r[1], top_r[2]});
        int mid_max = *std::max_element(middle_r.begin(), middle_r.end());
        if (mid_max <= top_max) return false;
      }
    }
  }
  return true;
}

std::pair<float, bool> score_board_from_ints(py::array_t<int16_t> bottom,
                                             py::array_t<int16_t> middle,
                                             py::array_t<int16_t> top){
  auto b = bottom.unchecked<1>();
  auto m = middle.unchecked<1>();
  auto t = top.unchecked<1>();
  if (b.shape(0)!=5 || m.shape(0)!=5 || t.shape(0)!=3) throw std::runtime_error("invalid sizes");
  std::array<int,5> br{}, bs{};
  std::array<int,5> mr{}, ms{};
  std::array<int,3> tr{};
  for (int i=0;i<5;++i){ br[i]=rank_from_int(b(i)); bs[i]=suit_from_int(b(i)); }
  for (int i=0;i<5;++i){ mr[i]=rank_from_int(m(i)); ms[i]=suit_from_int(m(i)); }
  for (int i=0;i<3;++i){ tr[i]=rank_from_int(t(i)); }
  bool valid = validate_board(br,bs,mr,ms,tr);
  if (!valid) {
    // Use canonical foul penalty from Rules.hpp so Python and C++ stay in sync.
    return { -static_cast<float>(ofc::rules::FOUL_PENALTY), true };
  }
  int roy = royalties_five(br,bs,false) + royalties_five(mr,ms,true) + std::max(top_pair_royalty(tr), top_trips_royalty(tr));
  return {static_cast<float>(roy), false};
}


