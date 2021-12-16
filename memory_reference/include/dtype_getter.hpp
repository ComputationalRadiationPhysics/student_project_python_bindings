#pragma once

template<typename TData> std::string get_dtype(){ return "C++ type not implemented";}
template<> std::string get_dtype<short int>(){ return "int16";}
template<> std::string get_dtype<int>(){ return "int32";}
template<> std::string get_dtype<long long int>(){ return "int64";}
template<> std::string get_dtype<std::uint16_t>(){ return "uint16";}
template<> std::string get_dtype<std::uint32_t>(){ return "uint32";}
template<> std::string get_dtype<std::uint64_t>(){ return "uint64";}
template<> std::string get_dtype<float>(){ return "float32";}
template<> std::string get_dtype<double>(){ return "float64";}
template<> std::string get_dtype<std::complex<float>>(){ return "complex64";}
template<> std::string get_dtype<std::complex<double>>(){ return "complex128";}