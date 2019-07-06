//
// Created by yche on 8/8/18.
//

#ifndef SCAN_QUERY_UTIL_H
#define SCAN_QUERY_UTIL_H

#include <string>

#include <iomanip>
#include <locale>
#include <sstream>

using namespace std;

template<class T>
std::string FormatWithCommas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}


#endif //SCAN_QUERY_UTIL_H
