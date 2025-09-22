// This is a C++ header file for the genetic functions

#ifndef GENETIC_UTILS_H
#define GENETIC_UTILS_H

#include <string>
#include <unordered_map>

namespace {
// Genetic code mapping for translation - inside anonymous namespace for internal linkage
static const std::unordered_map<std::string, char> GENETIC_CODE = {
    {"TTT", 'F'}, {"TTC", 'F'}, {"TTA", 'L'}, {"TTG", 'L'},
    {"CTT", 'L'}, {"CTC", 'L'}, {"CTA", 'L'}, {"CTG", 'L'},
    {"ATT", 'I'}, {"ATC", 'I'}, {"ATA", 'I'}, {"ATG", 'M'},
    {"GTT", 'V'}, {"GTC", 'V'}, {"GTA", 'V'}, {"GTG", 'V'},
    {"TCT", 'S'}, {"TCC", 'S'}, {"TCA", 'S'}, {"TCG", 'S'},
    {"CCT", 'P'}, {"CCC", 'P'}, {"CCA", 'P'}, {"CCG", 'P'},
    {"ACT", 'T'}, {"ACC", 'T'}, {"ACA", 'T'}, {"ACG", 'T'},
    {"GCT", 'A'}, {"GCC", 'A'}, {"GCA", 'A'}, {"GCG", 'A'},
    {"TAT", 'Y'}, {"TAC", 'Y'}, {"TAA", '*'}, {"TAG", '*'},
    {"CAT", 'H'}, {"CAC", 'H'}, {"CAA", 'Q'}, {"CAG", 'Q'},
    {"AAT", 'N'}, {"AAC", 'N'}, {"AAA", 'K'}, {"AAG", 'K'},
    {"GAT", 'D'}, {"GAC", 'D'}, {"GAA", 'E'}, {"GAG", 'E'},
    {"TGT", 'C'}, {"TGC", 'C'}, {"TGA", '*'}, {"TGG", 'W'},
    {"CGT", 'R'}, {"CGC", 'R'}, {"CGA", 'R'}, {"CGG", 'R'},
    {"AGT", 'S'}, {"AGC", 'S'}, {"AGA", 'R'}, {"AGG", 'R'},
    {"GGT", 'G'}, {"GGC", 'G'}, {"GGA", 'G'}, {"GGG", 'G'}
};
}  // End anonymous namespace

// Function to translate DNA sequence to amino acid sequence
inline std::string translate(const std::string& seq) {
    std::string result;
    size_t length = seq.length();
    
    // Reserve space to avoid reallocations
    result.reserve(length / 3 + 1);
    
    for (size_t i = 0; i + 2 < length; i += 3) {
        std::string codon = seq.substr(i, 3);
        auto it = GENETIC_CODE.find(codon);
        
        if (it != GENETIC_CODE.end()) {
            char amino_acid = it->second;
            if (amino_acid != '*') {  // Skip stop codons
                result.push_back(amino_acid);
            }
        } else {
            // Handle invalid codon (e.g., containing 'N')
            result.push_back('X');
        }
    }
    
    return result;
}

#endif // GENETIC_UTILS_H