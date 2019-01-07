// ising.cpp
// Tensor renormalization group (TRG) for classical 2D ising model
// uses ITensor (http://itensor.org)

#include "itensor/all.h"
#include "ising.h"

int main(int argc, char * argv[]) {
    // simulation parameters
    // dim = bond dimension of physical configuration space (d=2 for Ising spins)
    // num_iter = number of TRG passes to perform
    // 
    int dim = 2;
    int num_iter = 6;
    int dim_trunc = 32;
    double K;
    if (argc > 1)
        K = std::atof(argv[1]);
    else
        K = 1.0;
    
    auto r0 = itensor::Index("r", dim, itensor::Atype);
    auto l0 = itensor::prime(r0);
    auto u0 = itensor::Index("u", dim, itensor::Btype);
    auto d0 = itensor::prime(u0);

    auto T = itensor::ITensor(r0, u0, l0, d0);
    
    // lambda for spin variables
    // maps index to spin: 0 -> -1, 1 -> 1
    auto sp = [](int q) { return 2*q-1; };

    // generate the initial tensor
    for(auto i : itensor::range(dim))
        for(auto j : itensor::range(dim))
            for(auto k : itensor::range(dim))
                for(auto m : itensor::range(dim))
    {
        double t = 0.5*(1.0 + 1.0*sp(i)*sp(j)*sp(k)*sp(m));
        double p = exp(0.5*K*(sp(i) + sp(j) + sp(k) + sp(m)));
        T.set(r0(i + 1), u0(j + 1), l0(k + 1), d0(m + 1), t*p);
    }
    itensor::PrintData(T);

    // iterative renormalization scheme
    // two steps: 
    //  1) factor tensors using SVD (two different ways, to avoid gauge ambiguity)
    //  2) contract tensors to get a new "initial" tensor at longer wavelength
    for (auto iter : itensor::range(num_iter)) {

            auto r = itensor::findtype(T, itensor::Atype);
            auto u = itensor::findtype(T, itensor::Btype);
            auto l = itensor::prime(r);
            auto d = itensor::prime(u);

            // A type factorization
            auto S1 = itensor::ITensor(u, l, itensor::prime(r, 2));
            auto S3 = itensor::ITensor(d, r, itensor::prime(l, 2));
            itensor::factor(T, S1, S3, {"Minm", dim*dim,
                                        "Maxm", dim_trunc,
                                        "ShowEigs", true,
                                        "IndexName", "a",
                                        "IndexType", itensor::Atype});
            auto a = itensor::commonIndex(S1, S3);
            
            // B type factorization
            auto S2 = itensor::ITensor(l, d, itensor::prime(u, 2));
            auto S4 = itensor::ITensor(r, u, itensor::prime(d, 2));
            itensor::factor(T, S2, S4, {"Minm", dim*dim,
                                        "Maxm", dim_trunc,
                                        "ShowEigs", true,
                                        "IndexName", "b",
                                        "IndexType", itensor::Btype});
            auto b = itensor::commonIndex(S2, S4);

            S3.prime(a);
            S4.prime(b);

            // contract
            T = S1 * S2 * S3 * S4;
    }

    auto rN = itensor::findtype(T, itensor::Atype);
    auto uN = itensor::findtype(T, itensor::Btype);
    auto lN = itensor::prime(rN);
    auto dN = itensor::prime(uN);

    // delta tensors for contraction
    auto Trx = itensor::delta(rN, lN);
    auto Try = itensor::delta(uN, dN);

    // trace over the top-level tensor to get parition function Z
    auto Z = (Trx*T*Try).real();

    std::cout << "##########################" << std::endl;
    std::cout << "Coupling constant k = " << K;
    std::cout << ", Num. iterations " << num_iter << std::endl;
    std::cout << "Partition function, Z = " << Z << std::endl;
    itensor::Real Ns = pow(2, num_iter);
    itensor::printfln("Free energy per site, log(Z)/Ns = %.12f", log(Z)/Ns);
    return 0;
}


