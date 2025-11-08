#pragma once

#include <array>

#include "../types.h"
#include "../tensor.h"

namespace Ember::chess {
    enum Color {
        BLACK,
        WHITE
    };

    enum PieceType : i8 {
        PAWN,
        KNIGHT,
        BISHOP,
        ROOK,
        QUEEN,
        KING,
        NO_PIECE_TYPE
    };

    // clang-format off
    enum Square : i8 {
        a1, b1, c1, d1, e1, f1, g1, h1,
        a2, b2, c2, d2, e2, f2, g2, h2,
        a3, b3, c3, d3, e3, f3, g3, h3,
        a4, b4, c4, d4, e4, f4, g4, h4,
        a5, b5, c5, d5, e5, f5, g5, h5,
        a6, b6, c6, d6, e6, f6, g6, h6,
        a7, b7, c7, d7, e7, f7, g7, h7,
        a8, b8, c8, d8, e8, f8, g8, h8,
        NO_SQUARE
    };

    enum Direction : int {
        NORTH = 8,
        NORTH_EAST = 9,
        EAST = 1,
        SOUTH_EAST = -7,
        SOUTH = -8,
        SOUTH_WEST = -9,
        WEST = -1,
        NORTH_WEST = 7,
        NORTH_NORTH = 16,
        SOUTH_SOUTH = -16
    };

    enum File : int {
        FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H
    };
    enum Rank : int {
        RANK1, RANK2, RANK3, RANK4, RANK5, RANK6, RANK7, RANK8
    };

    constexpr u64 MASK_FILE[8] = {
      0x101010101010101, 0x202020202020202, 0x404040404040404, 0x808080808080808, 0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
    };
    constexpr u64 MASK_RANK[8] = {
        0xff, 0xff00, 0xff0000, 0xff000000, 0xff00000000, 0xff0000000000, 0xff000000000000, 0xff00000000000000
    };

    //Inverts the color (WHITE -> BLACK) and (BLACK -> WHITE)
    constexpr Color operator~(const Color c) { return static_cast<Color>(c ^ 1); }

    inline Square& operator++(Square& s) { return s = static_cast<Square>(static_cast<i8>(s) + 1); }
    inline Square& operator--(Square& s) { return s = static_cast<Square>(static_cast<i8>(s) - 1); }
    constexpr Square operator+(const Square s, const Direction d) { return static_cast<Square>(static_cast<i8>(s) + static_cast<i8>(d)); }
    constexpr Square operator-(const Square s, const Direction d) { return static_cast<Square>(static_cast<i8>(s) - static_cast<i8>(d)); }
    inline Square& operator+=(Square& s, const Direction d) { return s = s + d; }
    inline Square& operator-=(Square& s, const Direction d) { return s = s - d; }
    //clang-format on

    enum MoveType {
        STANDARD_MOVE = 0, EN_PASSANT = 0x4000, CASTLE = 0x8000, PROMOTION = 0xC000
    };

    constexpr std::array<Square, 4> ROOK_CASTLE_END_SQ = { d8, f8, d1, f1 };
    constexpr std::array<Square, 4> KING_CASTLE_END_SQ = { c8, g8, c1, g1 };

    class Move;

    struct Board {
        // Index is based on square, returns the piece type
        std::array<PieceType, 64> mailbox;
        // Indexed pawns, knights, bishops, rooks, queens, king
        std::array<u64, 6> byPieces;
        // Index is based on color, so black is colors[0]
        std::array<u64, 2> byColor;

        // En passant square
        Square epSquare;
        // Index KQkq
        std::array<Square, 4> castling;

        Color stm;

        usize halfMoveClock;
        usize fullMoveClock;

       private:
        void placePiece(Color c, PieceType pt, int sq);
        void removePiece(Color c, PieceType pt, int sq);
        void removePiece(Color c, int sq);
        void resetMailbox();

        void setCastlingRights(Color c, Square sq, bool value);
        void unsetCastlingRights(Color c);

        Square castleSq(Color c, bool kingside) const;

       public:
        u8 count(PieceType pt) const;

        u64 pieces() const;
        u64 pieces(Color c) const;
        u64 pieces(PieceType pt) const;
        u64 pieces(Color c, PieceType pt) const;
        u64 pieces(PieceType pt1, PieceType pt2) const;
        u64 pieces(Color c, PieceType pt1, PieceType pt2) const;

        void loadFromFEN(std::string fen);

        char getPieceAt(i8 i) const;

        PieceType getPiece(i8 sq) const;
        bool      isCapture(Move m) const;

        std::vector<float> asInputLayer() const;

        void move(Move m);

        friend std::ostream& operator<<(std::ostream& os, const Board& board);
    };
}
