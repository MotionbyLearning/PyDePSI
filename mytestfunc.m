function c = mytestfunc()
    load ('test.mat');
    c = calculateMatrixMult(a,b);
    save ('result.mat', 'c');
end