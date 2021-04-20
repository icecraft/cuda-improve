

# include <stdio.h>
# include <unistd.h>


const maxBase36N = 35;

static int base36 = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                    97, 98, 99, 100, 101, 102, 103, 104, 105,
                    106, 107, 108, 109, 110, 111, 112, 113, 114,
                    115, 116, 117, 118, 119, 120, 121, 122};

struct Base36 {
    int n;
    int dataIndex[];
    int data[];
};

bool isValidBase36 (struct Base36 *s ) {
     if 0 >= s-> n {
         return false;
     }
     for (int i = 0; i < n; i ++) {
         if 0 > s->data[i] || s->data[i] > maxBase36N {
             return false
         }
     }
     return true;
}

int incr(struct Base36 *s) {
    int carry = 1;
    int sum = 0;
    int div;
    for (int i = 0; i < s.n; i++) {
        sum = s->dataIndex[i] + carry;
        div = sum / 36;
        carry = sum % 36;
        s->dataIndex[i] = div;
        s->data[i] = base36[div];
    }
    return carry
}


int main(int argc, char* argv[]) {
    printf("hello world");
}