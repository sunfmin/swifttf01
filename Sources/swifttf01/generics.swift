
func swapTwoValues<T>(_ a: inout T, _ b: inout T) {
    let temporaryA = a
    a = b
    b = temporaryA
}


protocol Ordered where Self : Comparable {
    func less(other: Self) -> Bool
}


func binarySearch<T: Ordered>(sortedKeys: [T], forKey k: T) -> Int {
    var lo = 0, hi = sortedKeys.count
    while hi > lo {
        let mid = lo + (hi - lo) / 2
        if sortedKeys[mid].less(other: k) {
            lo = mid + 1
        } else {
            hi = mid
        }
    }
    return lo
}

extension Comparable {
    func less(other: Self) -> Bool {
        return self < other
    }
}

extension Int: Ordered {}
extension String: Ordered {}
//extension Double: Ordered {}


struct Stack<E> {
    var items = [E]()
    mutating func push(_ item: E) {
        items.append(item)
    }
    mutating func pop() -> E {
        return items.removeLast()
    }


    mutating func append(_ item: E) {
        self.push(item)
    }

    var count: Int {
        return items.count
    }

    subscript(i: Int) -> Item {
        return items[i]
    }

}

extension Stack {
    var top: E? {
        return items.isEmpty ? nil : items.last
    }
}

protocol Container {
    associatedtype Item
    mutating func append(_ item: Item)
    var count: Int {get}
    subscript(i: Int) -> Item { get }

}

extension Stack: Container {}
extension Array: Container {}

func findIndex<T: Equatable>(of value: T, in array: [T]) -> Int? {
    for (i, v) in array.enumerated() {
        if v == value {
            return i
        }
    }
    return nil
}

func printCount<T: Container>(of container: T) {
    print("printCount \(container.count)")
}


func allItemsTheSame<C1: Container, C2: Container>(_ c1: C1, _ c2: C2) -> Bool
        where C1.Item == C2.Item, C1.Item : Equatable {
    if c1.count != c2.count {
        return false
    }
    return true
}

extension Stack where E: Equatable {
    func isTop(_ item: E) -> Bool {
        guard let topItem = items.last else {
            return false
        }
        return topItem == item
    }
}

struct NotEq {}

extension Container where Item == Double {
    func average() -> Double {
        var sum = 0.0
        for index in 0..<count {
            sum += self[index]
        }
        return sum / Double(count)
    }
}


func doit(closure: (_ v: String) -> Void) {
    closure("Hello")
}



func runGen() {
    doit(closure: { v in
        print("doit \(v)")
    })

    doit() { v in
        print("doit trailing \(v)")
    }

    var someInt = 3
    var anotherInt = 107
    swapTwoValues(&someInt, &anotherInt)
    print("someInt is now \(someInt), and anotherInt is now \(anotherInt)")
    var someString = "Hello"
    var anotherString = "World"
    swapTwoValues(&someString, &anotherString)
    swap(&someString, &anotherString)
    print("someString is now \(someString), and anotherString is now \(anotherString)")

    if let i = findIndex(of: 4, in: [1, 2, 3, 123]) {
        print("findIndex \(i)")
    }

    var s = Stack<Int>()
    s.push(10)
    s.push(12)
    printCount(of: s)
    printCount(of: [1, 2, 3])
    print("allItemsTheSame \(allItemsTheSame(s, [1, 2]))")

    print(s.pop())
    if let v = s.top {
        print("top \(v)")
    }


    var noteqs = Stack<NotEq>()
    noteqs.push(NotEq())
    print("noteqs count \(noteqs.count)")
//    noteqs.isTop(nil)

    var ints = Stack<Double>()
    ints.push(1.0)
    ints.push(3.0)
    print("ints is Top", ints.isTop(1), ints.average())

    print(3.14.less(other: 5.0))
    print(binarySearch(sortedKeys: [1, 2, 3, 4, 5], forKey: 4));
    print(binarySearch(sortedKeys: ["1", "2", "3", "4", "5"], forKey: "1"));
//print(binarySearch(sortedKeys: [1.0, 2.0, 3.0, 4.0, 5.0], forKey: 5.0));
}
