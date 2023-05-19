function c = mytestfunc()
    load ('test.mat');
    c = calculateMatrixMult(a,b);
    savefast ('result.mat', 'c');
end