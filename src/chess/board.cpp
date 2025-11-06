#include "board.h"

#include "../util.h"

#include <sstream>
#include <random>
#include <bit>

using std::popcount;

#define ctzll(x) std::countr_zero(x)

namespace Ember::chess {
    // Helpers and util functions

    constexpr Square toSquare(const Rank rank, const File file) { return static_cast<Square>((static_cast<int>(rank) << 3) | file); }

    // Takes square (h8) and converts it into a bitboard index (64)
    constexpr Square parseSquare(const std::string_view square) { return static_cast<Square>((square.at(1) - '1') * 8 + (square.at(0) - 'a')); }

    bool readBit(const u64 bb, const i8 sq) { return (1ULL << sq) & bb; }

    template<bool value>
    void setBit(auto& bitboard, const usize index) {
        assert(index <= sizeof(bitboard) * 8);
        if constexpr (value)
            bitboard |= (1ULL << index);
        else
            bitboard &= ~(1ULL << index);
    }

    Square getLSB(auto bb) {
        assert(bb > 0);
        return static_cast<Square>(ctzll(bb));
    }

    Square popLSB(auto& bb) {
        assert(bb > 0);
        const Square sq = getLSB(bb);
        bb &= bb - 1;
        return sq;
    }

    template<int dir>
    u64 shift(const u64 bb) {
        return dir > 0 ? bb << dir : bb >> -dir;
    }

    u64 shift(const int dir, const u64 bb) { return dir > 0 ? bb << dir : bb >> -dir; }

    constexpr u8 castleIndex(const Color c, const bool kingside) { return c == WHITE ? (kingside ? 3 : 2) : (kingside ? 1 : 0); }

    constexpr Square flipRank(Square s) { return Square(s ^ 0b111000); }
    constexpr Square flipFile(Square s) { return Square(s ^ 0b000111); }



    // Encodes a chess move
    class Move {
        u16 move;

    public:
        constexpr Move()  = default;
        constexpr ~Move() = default;

        constexpr Move(const u8 startSquare, const u8 endSquare, const MoveType flags = STANDARD_MOVE) {
            move = startSquare | flags;
            move |= endSquare << 6;
        }

        constexpr Move(const u8 startSquare, const u8 endSquare, const PieceType promo) {
            move = startSquare | PROMOTION;
            move |= endSquare << 6;
            move |= (promo - 1) << 12;
        }

        Move(std::string strIn, Board& board);

        constexpr static Move null() { return Move(a1, a1); }

        std::string toString() const;

        Square from() const { return static_cast<Square>(move & 0b111111); }
        Square to() const { return static_cast<Square>((move >> 6) & 0b111111); }

        MoveType typeOf() const { return static_cast<MoveType>(move & 0xC000); }

        PieceType promo() const {
            assert(typeOf() == PROMOTION);
            return static_cast<PieceType>(((move >> 12) & 0b11) + 1);
        }

        bool isNull() const { return *this == null(); }

        bool operator==(const Move other) const { return move == other.move; }

        friend std::ostream& operator<<(std::ostream& os, const Move& m) {
            os << m.toString();
            return os;
        }
    };


    // Returns the piece on a square as a character
    char Board::getPieceAt(const i8 sq) const {
        assert(sq >= 0);
        assert(sq < 64);
        if (getPiece(sq) == NO_PIECE_TYPE)
            return ' ';
        constexpr char whiteSymbols[] = { 'P', 'N', 'B', 'R', 'Q', 'K' };
        constexpr char blackSymbols[] = { 'p', 'n', 'b', 'r', 'q', 'k' };
        if (((1ULL << sq) & byColor[WHITE]) != 0)
            return whiteSymbols[getPiece(sq)];
        return blackSymbols[getPiece(sq)];
    }

    void Board::placePiece(const Color c, const PieceType pt, const int sq) {
        assert(sq >= 0);
        assert(sq < 64);

        auto& BB = byPieces[pt];

        assert(!readBit(BB, sq));

        BB ^= 1ULL << sq;
        byColor[c] ^= 1ULL << sq;

        mailbox[sq] = pt;
    }

    void Board::removePiece(Color c, PieceType pt, int sq) {
        assert(sq >= 0);
        assert(sq < 64);

        auto& BB = byPieces[pt];

        assert(readBit(BB, sq));

        BB ^= 1ULL << sq;
        byColor[c] ^= 1ULL << sq;

        mailbox[sq] = NO_PIECE_TYPE;
    }

    void Board::removePiece(Color c, int sq) {
        assert(sq >= 0);
        assert(sq < 64);

        auto& BB = byPieces[getPiece(sq)];

        assert(readBit(BB, sq));

        BB ^= 1ULL << sq;
        byColor[c] ^= 1ULL << sq;

        mailbox[sq] = NO_PIECE_TYPE;
    }

    void Board::resetMailbox() {
        mailbox.fill(NO_PIECE_TYPE);
        for (u8 i = 0; i < 64; i++) {
            PieceType& sq   = mailbox[i];
            const u64  mask = 1ULL << i;
            if (mask & pieces(PAWN))
                sq = PAWN;
            else if (mask & pieces(KNIGHT))
                sq = KNIGHT;
            else if (mask & pieces(BISHOP))
                sq = BISHOP;
            else if (mask & pieces(ROOK))
                sq = ROOK;
            else if (mask & pieces(QUEEN))
                sq = QUEEN;
            else if (mask & pieces(KING))
                sq = KING;
        }
    }

    void Board::setCastlingRights(const Color c, const Square sq, const bool value) { castling[castleIndex(c, ctzll(pieces(c, KING)) < sq)] = (value == false ? NO_SQUARE : sq); }

    void Board::unsetCastlingRights(const Color c) { castling[castleIndex(c, true)] = castling[castleIndex(c, false)] = NO_SQUARE; }

    Square Board::castleSq(const Color c, const bool kingside) const { return castling[castleIndex(c, kingside)]; }

    u8 Board::count(const PieceType pt) const { return popcount(pieces(pt)); }

    u64 Board::pieces() const { return byColor[WHITE] | byColor[BLACK]; }
    u64 Board::pieces(const Color c) const { return byColor[c]; }
    u64 Board::pieces(const PieceType pt) const { return byPieces[pt]; }
    u64 Board::pieces(const Color c, const PieceType pt) const { return byPieces[pt] & byColor[c]; }
    u64 Board::pieces(const PieceType pt1, const PieceType pt2) const { return byPieces[pt1] | byPieces[pt2]; }
    u64 Board::pieces(const Color c, const PieceType pt1, const PieceType pt2) const { return (byPieces[pt1] | byPieces[pt2]) & byColor[c]; }

    // Load a board from the FEN
    void Board::loadFromFEN(const std::string fen) {
        // Clear all squares
        byPieces.fill(0);
        byColor.fill(0);

        const std::vector<std::string> tokens = split(fen, ' ');

        const std::vector<std::string> rankTokens = split(tokens[0], '/');

        int currIdx = 56;

        constexpr char whitePieces[6] = { 'P', 'N', 'B', 'R', 'Q', 'K' };
        constexpr char blackPieces[6] = { 'p', 'n', 'b', 'r', 'q', 'k' };

        for (const std::string& rank : rankTokens) {
            for (const char c : rank) {
                if (isdigit(c)) {  // Empty squares
                    currIdx += c - '0';
                    continue;
                }
                for (int i = 0; i < 6; i++) {
                    if (c == whitePieces[i]) {
                        setBit<1>(byPieces[i], currIdx);
                        setBit<1>(byColor[WHITE], currIdx);
                        break;
                    }
                    if (c == blackPieces[i]) {
                        setBit<1>(byPieces[i], currIdx);
                        setBit<1>(byColor[BLACK], currIdx);
                        break;
                    }
                }
                currIdx++;
            }
            currIdx -= 16;
        }

        if (tokens[1] == "w")
            stm = WHITE;
        else
            stm = BLACK;

        castling.fill(NO_SQUARE);
        if (tokens[2].find('-') == std::string::npos) {
            // Standard FEN and maybe XFEN later
            if (tokens[2].find('K') != std::string::npos)
                castling[castleIndex(WHITE, true)] = h1;
            if (tokens[2].find('Q') != std::string::npos)
                castling[castleIndex(WHITE, false)] = a1;
            if (tokens[2].find('k') != std::string::npos)
                castling[castleIndex(BLACK, true)] = h8;
            if (tokens[2].find('q') != std::string::npos)
                castling[castleIndex(BLACK, false)] = a8;

            // FRC FEN
            if (std::tolower(tokens[2][0]) >= 'a' && std::tolower(tokens[2][0]) <= 'h') {
                for (char token : tokens[2]) {
                    const auto file = static_cast<File>(std::tolower(token) - 'a');

                    if (std::isupper(token))
                        setCastlingRights(WHITE, toSquare(RANK1, file), true);
                    else
                        setCastlingRights(BLACK, toSquare(RANK8, file), true);
                }
            }
        }

        if (tokens[3] != "-")
            epSquare = parseSquare(tokens[3]);
        else
            epSquare = NO_SQUARE;

        halfMoveClock = tokens.size() > 4 ? (stoi(tokens[4])) : 0;
        fullMoveClock = tokens.size() > 5 ? (stoi(tokens[5])) : 1;

        resetMailbox();
    }

    // Return the type of the piece on the square
    PieceType Board::getPiece(const i8 sq) const {
        assert(sq >= 0);
        assert(sq < 64);
        return mailbox[sq];
    }

    bool Board::isCapture(const Move m) const { return ((1ULL << m.to() & pieces(~stm)) || m.typeOf() == EN_PASSANT); }

    std::vector<float> Board::asInputLayer() const {
        const auto getFeature = [this](const Color pieceColor, const Square square) {
            const bool enemy       = stm != pieceColor;
            const int  squareIndex = (stm == BLACK) ? flipRank(square) : static_cast<int>(square);

            return enemy * 64 * 6 + getPiece(square) * 64 + squareIndex;
        };

        std::vector<float> res(2 * 6 * 64);

        u64 whitePieces = pieces(WHITE);
        u64 blackPieces = pieces(BLACK);

        while (whitePieces) {
            const Square sq = popLSB(whitePieces);

            const usize feature = getFeature(WHITE, sq);

            res[feature] = true;
        }

        while (blackPieces) {
            const Square sq = popLSB(blackPieces);

            const usize feature = getFeature(BLACK, sq);

            res[feature] = true;
        }

        return res;
    }

    void Board::move(const Move m) {
        epSquare       = NO_SQUARE;
        Square    from = m.from();
        Square    to   = m.to();
        MoveType  mt   = m.typeOf();
        PieceType pt   = getPiece(from);
        PieceType toPT = NO_PIECE_TYPE;

        removePiece(stm, pt, from);
        if (isCapture(m)) {
            toPT          = getPiece(to);
            halfMoveClock = 0;
            if (mt != EN_PASSANT) {
                removePiece(~stm, toPT, to);
            }
        }
        else {
            if (pt == PAWN)
                halfMoveClock = 0;
            else
                halfMoveClock++;
        }

        switch (mt) {
            case STANDARD_MOVE:
                placePiece(stm, pt, to);
                if (pt == PAWN && (to + 16 == from || to - 16 == from)
                    && (pieces(~stm, PAWN) & (shift<EAST>((1ULL << to) & ~MASK_FILE[FILE_H]) | shift<WEST>((1ULL << to) & ~MASK_FILE[FILE_A]))))  // Only set EP square if it could be taken
                        epSquare = static_cast<Square>(stm == WHITE ? from + NORTH : from + SOUTH);
                break;
            case EN_PASSANT:
                removePiece(~stm, PAWN, to + (stm == WHITE ? SOUTH : NORTH));
                placePiece(stm, pt, to);
                break;
            case CASTLE:
                assert(getPiece(to) == ROOK);
                removePiece(stm, ROOK, to);
                if (stm == WHITE) {
                    if (from < to) {
                        placePiece(stm, KING, g1);
                        placePiece(stm, ROOK, f1);
                    }
                    else {
                        placePiece(stm, KING, c1);
                        placePiece(stm, ROOK, d1);
                    }
                }
                else {
                    if (from < to) {
                        placePiece(stm, KING, g8);
                        placePiece(stm, ROOK, f8);
                    }
                    else {
                        placePiece(stm, KING, c8);
                        placePiece(stm, ROOK, d8);
                    }
                }
                break;
            case PROMOTION:
                placePiece(stm, m.promo(), to);
                break;
        }

        assert(popcount(pieces(WHITE, KING)) == 1);
        assert(popcount(pieces(BLACK, KING)) == 1);

        if (pt == ROOK) {
            const Square sq = castleSq(stm, from > ctzll(pieces(stm, KING)));
            if (from == sq)
                setCastlingRights(stm, from, false);
        }
        else if (pt == KING)
            unsetCastlingRights(stm);
        if (toPT == ROOK) {
            const Square sq = castleSq(~stm, to > ctzll(pieces(~stm, KING)));
            if (to == sq)
                setCastlingRights(~stm, to, false);
        }

        stm = ~stm;

        fullMoveClock += stm == WHITE;
    }
}