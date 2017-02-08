//
// A simple-minded parser that takes commands like
//
//   command a b c
//
// and calls the appropriate function with arguments a,b,c
//
// Andrew Pontzen 2013

#ifndef _PARSER_HPP
#define _PARSER_HPP

#include <memory>
#include <sstream>
#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <iostream>
#include <exception>
#include <utility>
#include <functional>
#include <cctype>

template<typename Rtype>
class Dispatch {
private:

  class DispatchError : public std::runtime_error {
  public:
    DispatchError(const char *x) : runtime_error(x) { }

  };

  struct Base {
    virtual ~Base() { }
  };


  template<class R, class... Args>
  struct Func : Base {
    std::function<R(Args...)> f;
  };

  typedef std::function<Rtype(const Base *, std::istream &, std::ostream *)> callerfn;

  std::map<std::string,
    std::pair<std::shared_ptr<Base>, callerfn> > _map;

  static void consume_comments(std::istream &input_stream) {
    std::string s;
    input_stream >> s;
    if (s[0] == '#' || s[0] == '%') {
      while (!input_stream.eof()) {
        input_stream >> s;
      }
    } else {
      if (s.size() != 0)
        throw DispatchError("Too many arguments");
    }
  }


  template<typename... Args>
  static Rtype call_function(const std::function<Rtype()> &f,
                             std::istream &input_stream,
                             std::ostream *output_stream) {
    consume_comments(input_stream);
    return f();
  }

  template<typename T1>
  static Rtype call_function(const std::function<Rtype(T1)> &f,
                             std::istream &input_stream,
                             std::ostream *output_stream) {
    T1 t1;

    if (input_stream.eof())
      throw DispatchError("Insufficient arugments (expected 1, got 0)");
    input_stream >> t1;

    consume_comments(input_stream);

    if (output_stream != nullptr)
      (*output_stream) << t1 << std::endl;

    return f(t1);
  }

  template<typename T1, typename T2>
  static Rtype call_function(const std::function<Rtype(T1, T2)> &f,
                             std::istream &input_stream,
                             std::ostream *output_stream) {

    if (input_stream.eof())
      throw DispatchError("Insufficient arugments (expected 2, got 0)");
    T1 t1;
    input_stream >> t1;
    if (input_stream.eof())
      throw DispatchError("Insufficient arugments (expected 2, got 1)");
    T2 t2;
    input_stream >> t2;
    consume_comments(input_stream);
    if (output_stream != nullptr)
      (*output_stream) << t1 << " " << t2 << std::endl;
    return f(t1, t2);
  }

  template<typename T1, typename T2, typename T3>
  static Rtype call_function(const std::function<Rtype(T1, T2, T3)> &f,
                             std::istream &input_stream,
                             std::ostream *output_stream) {

    if (input_stream.eof())
      throw DispatchError("Insufficient arugments (expected 3, got 0)");
    T1 t1;
    input_stream >> t1;
    if (input_stream.eof())
      throw DispatchError("Insufficient arugments (expected 3, got 1)");
    T2 t2;
    input_stream >> t2;
    if (input_stream.eof())
      throw DispatchError("Insufficient arugments (expected 3, got 2)");
    T3 t3;
    input_stream >> t3;
    consume_comments(input_stream);
    if (output_stream != nullptr)
      (*output_stream) << t1 << " " << t2 << " " << t3 << std::endl;
    return f(t1, t2, t3);
  }

  template<typename... Args>
  static Rtype unpack_and_call_function(const Base *fobj,
                                        std::istream &input_stream,
                                        std::ostream *output_stream) {

    // Turn the function back into the correct type (with argument information)
    // and call it using arguments on the input_stream

    auto pfunc = dynamic_cast<const Func<Rtype, Args...> *>(fobj);

    if (pfunc)
      return call_function<Args...>(pfunc->f, input_stream, output_stream);
    else
      throw DispatchError("Wrong type");
  }

  Rtype run(std::istream &input_stream, std::ostream *output_stream) {
    // Take a single line out of the input stream and try to execute it

    std::string s;
    input_stream >> s;

    if (s[0] == '#' || s[0] == '%') // comment
      return;

    decltype(_map.at(s).first.get()) function;
    decltype(_map.at(s).second) caller;

    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    // retrieve the function and something to call it with
    try {
      function = _map.at(s).first.get();
      caller = _map.at(s).second;
    } catch (std::out_of_range &e) {
      if (ignoreUnknownCommand)
        return Rtype();
      else
        throw DispatchError("Unknown command");
    }

    if (output_stream != nullptr)
      (*output_stream) << s << " ";

    // do it!
    return caller(function, input_stream, output_stream);
  }


  void run_loop(std::istream &input_stream, std::ostream *output_stream) {
    std::string str;
    std::stringstream ss;

    int line = 0;
    while (std::getline(input_stream, str)) {
      line += 1;
      if (str.size() == 0) continue;
      ss.str(str);
      try {
        run(ss, output_stream);
      } catch (std::exception &e) {
        std::cerr << "Error \"" << e.what() << "\" on line " << line << " (\"" << str << "\")" << std::endl;
        exit(1);
      }
      ss.clear();
    }

  }

public:
  bool ignoreUnknownCommand;

  Dispatch(bool ignoreUnknownCommand = false) : ignoreUnknownCommand(ignoreUnknownCommand) { }

  template<typename... Args>
  void add_route(const std::string &name,
                 const std::function<Rtype(Args...)> &function) {
    // Define a route that calls the given function

    auto pfunc = std::make_shared<Func<Rtype, Args...>>();
    auto pcaller = std::function<Rtype(const Base *, std::istream &, std::ostream *)>(
      &unpack_and_call_function<Args...>);
    pfunc->f = function;

    std::string lname(name);

    std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
    _map.insert(std::make_pair(lname, std::make_pair(pfunc, pcaller)));
  }

  Rtype run(std::string input) {
    std::istringstream ss(input);
    run(ss);
  }

  void run(std::istream &input_stream) {
    run(input_stream, nullptr);
  }

  void run(std::istream &input_stream, std::ostream &output_stream) {
    run(input_stream, &output_stream);
  }

  void run_loop(std::istream &input_stream) {
    run_loop(input_stream, nullptr);
  }

  void run_loop(std::istream &input_stream, std::ostream &output_stream) {
    run_loop(input_stream, &output_stream);
  }

};

template<typename Ctype, typename Rtype>
class InstanceDispatch;

template<typename Ctype, typename Rtype>
class ClassDispatch {
private:
  std::vector<std::function<void(InstanceDispatch<Ctype, Rtype> &)>> adderFunctions;
public:
  ClassDispatch() { }

  template<typename... Args>
  void add_class_route(const std::string &name, Rtype (Ctype::*f)(Args...)) {
    auto addcall = std::function<void(InstanceDispatch<Ctype, Rtype> &)>(
      [name, f](InstanceDispatch<Ctype, Rtype> &pDispatchObj) {
        pDispatchObj.add_class_route(name, f);
      });


    // make a lambda that adds the route to a specific object
    adderFunctions.emplace_back(addcall);
  }

  InstanceDispatch<Ctype, Rtype> specify_instance(Ctype &c) {
    auto id = InstanceDispatch<Ctype, Rtype>(c);
    for (auto fn : adderFunctions)
      fn(id);
    return id;

  }
};

template<typename Ctype, typename Rtype>
class InstanceDispatch : public Dispatch<Rtype> {
private:
  Ctype *pC;
  ClassDispatch<Ctype, Rtype> prototype;

public:
  InstanceDispatch(Ctype &c) : pC(&c) { }

  template<typename... Args>
  void add_class_route(const std::string &name, Rtype (Ctype::*f)(Args...)) {
    // make a lambda that performs the call
    auto call = [this, f](Args... input_args) { (pC->*f)(input_args...); };
    // add it as the route
    this->add_route(name, std::function<Rtype(Args...)>(call));
  }


};



/* TEST CODE --

class Stateful {
    double v=0;
public:
    void printout() {
        std::cerr << v << std::endl;
    }

    void add(double v1, double v2) {
        v=v1+v2;
    }

    void accum(double v1) {
        v+=v1;
    }
};

int main() {

    Stateful x;
    InstanceDispatch<Stateful, void> dp(x);

    dp.add_class_route("printout", &Stateful::printout);
    dp.add_class_route("add", &Stateful::add);
    dp.add_class_route("accum", &Stateful::accum);

    dp.run_loop(std::cin);



}

*/

#endif