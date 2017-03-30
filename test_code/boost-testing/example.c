/* example.c */

int Foo = 7;

int test_func(int n, int m);
int hello(int n);

int test_func(int n, int m) {
	Foo += 1;
	return hello(n) * m * Foo;
}

int hello(int n){
	if (n <= 1) {
		return 1;
	} else {
		return n * hello(n - 1);
	} 
}