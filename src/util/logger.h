/* BrushUSH
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef LOGGER_H
#define LOGGER_H

#include<iostream>
using namespace std;

namespace Brush {
namespace Util{
    
/*!
 * @class Logger
 * @brief Defines a multi level static logger.
 */
class Logger
{
public:
    
    /*!
        * @brief Initializes the logger instance.
        * @return A pointer to the logger instance.
        */
    static Logger* initLogger();
    
    /*!
        * @brief Destroys the logger instance.
        */
    static void destroy();

    /*!
        * @brief Sets the log level.
        * @param verbosity The log level to be set.
        */
    void set_log_level(int& verbosity);
    
    /*!
        * @brief Gets the current log level.
        * @return The current log level.
        */
    int get_log_level();
    
    /*!
        * @brief Prints a log message with verbosity control.
        * @param m The log message to be printed.
        * @param v The verbosity level of the log message.
        * @param sep The separator to be used between log messages.
        * @return The formatted log message.
        */
    string log(string m, int v, string sep="\n") const;
    
private:
    int verbosity; //!< The current log level.
    static Logger* instance; //!< The singleton instance of the logger.
};

static Logger &logger = *Logger::initLogger();

}
}
#endif
