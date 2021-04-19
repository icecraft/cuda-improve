

# include <stdio.h>
# include <unistd.h>



static int base10 = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57};
static int base26 = {97, 98, 99, 100, 101, 102, 103, 104, 105,
                    106, 107, 108, 109, 110, 111, 112, 113, 114,
                    115, 116, 117, 118, 119, 120, 121, 122};

struct Base36 {
    int n;
    int data[];
};


 isValidBase36 (struct Base36 *s ) {
    if 0 >= s->n {
        return false;
    }

}


int main(int argc, char* argv[]) {
    printf("hello world");
}